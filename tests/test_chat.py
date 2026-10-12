import pytest

from inferscope.chat import ManagedServer, Turn, parse_prometheus, token_counters


METRICS = """
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 59
# HELP llamacpp:tokens_predicted_total Number of generation tokens processed
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 28
llamacpp:request_success_total{request_type="chat"} 3
"""


def test_parse_prometheus_skips_labelled_series():
    m = parse_prometheus(METRICS)
    assert m["llamacpp:prompt_tokens_total"] == 59
    assert m["llamacpp:tokens_predicted_total"] == 28
    # A labelled series is skipped rather than collapsed into the parent key,
    # which would report one label set as if it were the total.
    assert "llamacpp:request_success_total" not in m


def test_token_counters():
    c = token_counters(METRICS)
    assert c == {"prompt_tokens": 59, "completion_tokens": 28}


def test_token_counters_missing_is_zero_not_invented():
    c = token_counters("# nothing here\n")
    assert c == {"prompt_tokens": 0, "completion_tokens": 0}


def test_turn_tok_per_s():
    t = Turn(completion_tokens=28, total_ms=2000.0)
    assert t.tok_per_s == pytest.approx(14.0)


def test_turn_tok_per_s_none_without_tokens_or_time():
    assert Turn(completion_tokens=0, total_ms=2000.0).tok_per_s is None
    assert Turn(completion_tokens=10, total_ms=0).tok_per_s is None
    assert Turn().tok_per_s is None


def test_managed_server_argv_binds_loopback():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081, ctx=4096)
    argv = s.argv()
    assert argv[0] == "/bin/llama-server"
    assert "--metrics" in argv
    assert "127.0.0.1" in argv
    assert argv[argv.index("--port") + 1] == "8081"
    assert argv[argv.index("-c") + 1] == "4096"
    assert s.base_url == "http://127.0.0.1:8081"


def test_managed_server_not_running_before_start():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081)
    assert s.running is False
    st = s.status()
    assert st["running"] is False and st["pid"] is None


def test_managed_server_stop_without_start():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081)
    assert s.stop() == {"stopped": False, "detail": "not running"}


def test_status_is_honest_about_trace_scope():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081)
    st = s.status()
    assert st["kernel_trace"] == "not capturing"
    assert st["per_turn_trace"] is False


def test_profile_mode_wraps_server_in_rocprofv3():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081,
                      profile=True, out_dir="/tmp/trace-out")
    argv = s.argv()
    assert argv[0] == "rocprofv3"
    assert "--kernel-trace" in argv
    # The server itself must remain the wrapped command.
    inner = s.server_argv()
    assert argv[-len(inner):] == inner


def test_profile_mode_requires_out_dir():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081, profile=True)
    try:
        s.argv()
    except ValueError as e:
        assert "out_dir" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ptrace_capable_returns_reason_not_crash():
    from inferscope.chat import ptrace_capable
    ok, why = ptrace_capable()
    assert isinstance(ok, bool)
    assert isinstance(why, str) and why

def test_port_in_use_detects_listener():
    import socket
    from inferscope.chat import port_in_use
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert port_in_use(port) is not None
    finally:
        s.close()
    # A closed port must not be reported as busy.
    assert port_in_use(port) is None


def test_start_refuses_port_already_in_use():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    srv = ManagedServer("/bin/llama-server", "/m/model.gguf", port)
    try:
        srv.start("/tmp/should-not-be-used.log")
        assert srv.proc is None
        assert "already in use" in srv.error
        assert "will not adopt" in srv.error
    finally:
        s.close()


def test_status_surfaces_error():
    s = ManagedServer("/bin/llama-server", "/m/model.gguf", 8081)
    s.error = "boom"
    assert s.status()["error"] == "boom"


def test_tunnel_status_is_tri_state_not_a_guess():
    """reachable must be True/False/None — never a fabricated False.

    The pod has no hairpin route to its own public URL, so an HTTP self-probe
    would always fail; status() therefore reports the frpc client's own state.
    """
    from inferscope import tunnel
    st = tunnel.status()
    assert st["reachable"] in (True, False, None)
    assert isinstance(st["detail"], str) and st["detail"]


def test_turn_exposes_reasoning_separately():
    """A reasoning model streams thinking before any content. If that is not
    surfaced, a short turn looks like the model said nothing."""
    t = Turn(reasoning="The user asks about GPUs.")
    d = t.as_dict()
    assert d["reasoning"] == "The user asks about GPUs."
    assert d["reasoning_only"] is True
    assert d["text"] == ""


def test_turn_with_content_is_not_reasoning_only():
    t = Turn(text="GPUs are fast.", reasoning="thinking...")
    assert t.as_dict()["reasoning_only"] is False


def test_reap_refuses_a_foreign_listener():
    """The orphan reaper must never touch a server InferScope does not own.

    The whole point of refusing to adopt a foreign server is that kernel
    tracing can only attach to a pid we spawned. If the reaper guessed wrong it
    would kill somebody else's inference job, so it requires positive evidence
    (our model path, or our working directory) and otherwise declines.
    """
    import socket
    from inferscope.chat import reap_own_orphan
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        # this listener is python3, not llama-server, and shares neither the
        # model path nor the app root -> left alone
        assert reap_own_orphan(port, app_root="/workspace/inferscope",
                               model_path="/workspace/models/other.gguf") is None
        assert s.fileno() >= 0          # still ours, still listening
        # a free port is a no-op
        assert reap_own_orphan(1, app_root="/", model_path="") is None
    finally:
        s.close()


def test_reap_recognises_our_own_cwd(monkeypatch):
    """Positive evidence: a llama-server whose cwd is this app is ours."""
    from inferscope import chat
    calls: list[tuple[int, int]] = []

    monkeypatch.setattr(chat, "port_in_use", lambda port: 4242)
    monkeypatch.setattr(chat, "_proc_cmdline",
                        lambda pid: "/workspace/build/llama-server -m /m/x.gguf --port 9")
    monkeypatch.setattr(chat, "_proc_cwd", lambda pid: "/workspace/inferscope/")
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    monkeypatch.setattr(chat.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    # make the post-kill check report the port as still busy so we exercise the
    # escalation to SIGKILL
    seq = iter([4242, 4242, None, None])
    monkeypatch.setattr(chat, "port_in_use", lambda port: next(seq, None))

    pid = chat.reap_own_orphan(8081, app_root="/workspace/inferscope", model_path="")
    assert pid == 4242
    assert calls, "expected a kill to be issued"
    assert calls[0][1] == chat.signal.SIGTERM


def test_reap_ignores_a_foreign_llama_server(monkeypatch):
    """A llama-server on our port but someone else's model and cwd is theirs."""
    from inferscope import chat
    killed: list[int] = []
    monkeypatch.setattr(chat, "port_in_use", lambda port: 777)
    monkeypatch.setattr(chat, "_proc_cmdline",
                        lambda pid: "/opt/llama-server -m /home/someone/model.gguf")
    monkeypatch.setattr(chat, "_proc_cwd", lambda pid: "/home/someone")
    monkeypatch.setattr(chat.os, "kill", lambda pid, sig: killed.append(pid))
    assert chat.reap_own_orphan(8081, app_root="/workspace/inferscope",
                                model_path="/workspace/models/Qwen3.8.gguf") is None
    assert not killed


def test_reap_recognises_another_engine_kind(monkeypatch):
    """The reaper must be generic across engine kinds.

    It only ever recognised ``llama-server``, so an orphaned ``whisper-server``
    — matching our model path exactly — was dismissed as "not an inference
    server" and left holding the ASR port forever.
    """
    from inferscope import chat
    killed: list[tuple[int, int]] = []
    seq = iter([4242, 4242, None, None])
    monkeypatch.setattr(chat, "port_in_use", lambda port: next(seq, None))
    monkeypatch.setattr(chat, "_proc_cmdline", lambda pid:
        "/workspace/build/whisper.cpp/build-hip/bin/whisper-server "
        "-m /workspace/models/whisper/ggml-base.en.bin --port 8082")
    monkeypatch.setattr(chat, "_proc_cwd", lambda pid: "/somewhere/else")
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    monkeypatch.setattr(chat.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    pid = chat.reap_own_orphan(8082, app_root="/workspace/inferscope",
                               model_path="/workspace/models/whisper/ggml-base.en.bin",
                               expect="whisper-server")
    assert pid == 4242
    assert killed and killed[0][1] == chat.signal.SIGTERM


def test_reap_will_not_kill_a_different_binary_on_our_model(monkeypatch):
    """Matching the model path is necessary but not sufficient: the process must
    also be the binary we would have started."""
    from inferscope import chat
    killed: list[int] = []
    monkeypatch.setattr(chat, "port_in_use", lambda port: 555)
    monkeypatch.setattr(chat, "_proc_cmdline", lambda pid:
        "/opt/someone-elses-tool -m /workspace/models/whisper/ggml-base.en.bin")
    monkeypatch.setattr(chat, "_proc_cwd", lambda pid: "/home/someone")
    monkeypatch.setattr(chat.os, "kill", lambda pid, sig: killed.append(pid))
    assert chat.reap_own_orphan(8082, app_root="/workspace/inferscope",
                                model_path="/workspace/models/whisper/ggml-base.en.bin",
                                expect="whisper-server") is None
    assert not killed
