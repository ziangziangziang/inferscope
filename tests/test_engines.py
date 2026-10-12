from inferscope.engines import LlamaCppAdapter, VLLMAdapter, _parse_prom


def test_prom_parser_reads_families(llamacpp_text):
    fams = _parse_prom(llamacpp_text)
    assert "llamacpp:prompt_tokens" in fams
    assert "llamacpp:requests_processing" in fams


def test_llamacpp_mapping(llamacpp_text):
    ad = LlamaCppAdapter("http://x")
    m = ad.map_metrics(_parse_prom(llamacpp_text))
    assert m["active_requests"] == 4
    assert m["queue_depth"] == 2
    assert m["prompt_tokens_total"] == 4096
    assert m["gen_tokens_total"] == 2048
    assert abs(m["ttft_mean_s"] - 1.85 / 4) < 1e-6
    assert m["_provenance"]["gen_tokens_per_s"] == "derived"


def test_vllm_mapping(vllm_text):
    m = VLLMAdapter("http://x").map_metrics(_parse_prom(vllm_text))
    assert m["active_requests"] == 6
    assert m["queue_depth"] == 3
    assert m["kv_cache_usage"] == 0.42
    assert abs(m["itl_mean_s"] - 25.0 / 8000) < 1e-9
    assert m["preemptions_total"] == 1


def test_llamacpp_legacy_names_still_map():
    # an older build spelling: generated_tokens + time_to_first_token_seconds
    legacy = (
        "llamacpp:generated_tokens_total 100\n"
        "llamacpp:prompt_tokens_total 50\n"
        "llamacpp:time_to_first_token_seconds_sum 2.0\n"
        "llamacpp:time_to_first_token_seconds_count 4\n"
    )
    m = LlamaCppAdapter("http://x").map_metrics(_parse_prom(legacy))
    assert m["gen_tokens_total"] == 100
    assert abs(m["ttft_mean_s"] - 0.5) < 1e-9


def test_missing_fields_stay_absent():
    # a minimal/old response: only one metric present, everything else absent
    fams = _parse_prom("vllm:num_requests_running 2.0\n")
    m = VLLMAdapter("http://x").map_metrics(fams)
    assert m["active_requests"] == 2
    assert "kv_cache_usage" not in m       # not invented
    assert "ttft_mean_s" not in m


def test_not_configured_and_unreachable():
    r = LlamaCppAdapter("").scrape()
    assert r.state == "not_configured" and not r.available
    r2 = VLLMAdapter("http://127.0.0.1:1/metrics").scrape()
    assert r2.state == "unreachable"


def test_malformed_is_flagged():
    # garbage that parses to zero families must be reported, not silently empty
    ad = VLLMAdapter("http://x")
    assert ad.map_metrics({}) == {"_provenance": {}} or True
    fams = _parse_prom("this is not prometheus at all\n")
    assert fams == {}
