from inferscope.kv import KVInputs, kv_cache_bytes, gqa_note, dtype_bytes


def test_baseline_formula_exact():
    # 2 * L * B * C * Hkv * D * bytes  = 2*32*1*2048*8*128*2 = 268,435,456 = 256 MiB
    r = kv_cache_bytes(KVInputs(n_layers=32, context=2048, batch=1,
                                kv_heads=8, head_dim=128, dtype="bfloat16"))
    assert r.bytes == 2 * 32 * 1 * 2048 * 8 * 128 * 2
    assert r.baseline_bytes == r.bytes


def test_gqa_reduces_footprint():
    mha = kv_cache_bytes(KVInputs(n_layers=4, context=100, kv_heads=8, head_dim=64)).bytes
    gqa = kv_cache_bytes(KVInputs(n_layers=4, context=100, kv_heads=2, head_dim=64)).bytes
    assert gqa * 4 == mha  # 8->2 heads is a 4x reduction
    assert "grouped-query" in gqa_note(8, 2)


def test_sliding_window_lowers_estimate():
    full = kv_cache_bytes(KVInputs(n_layers=8, context=8192, kv_heads=4, head_dim=128)).bytes
    swa = kv_cache_bytes(KVInputs(n_layers=8, context=8192, kv_heads=4, head_dim=128,
                                  sliding_window=2048)).bytes
    assert swa < full


def test_shared_prefix_saving():
    plain = kv_cache_bytes(KVInputs(n_layers=4, context=1000, batch=4, kv_heads=2, head_dim=64)).bytes
    shared = kv_cache_bytes(KVInputs(n_layers=4, context=1000, batch=4, kv_heads=2, head_dim=64,
                                     shared_prefix=500)).bytes
    assert shared < plain
    assert any("shared prefix" in n for n in shared_notes(shared, plain))


def shared_notes(a, b):
    return kv_cache_bytes(KVInputs(n_layers=4, context=1000, batch=4, kv_heads=2,
                                   head_dim=64, shared_prefix=500)).notes


def test_dtype_bytes():
    assert dtype_bytes("bfloat16") == 2
    assert dtype_bytes("int8") == 1
    assert dtype_bytes("q4_0") == 0.5


def test_validation_rejects_bad_input():
    assert KVInputs(n_layers=0, context=10).validate()
    assert KVInputs(n_layers=4, context=-1).validate()
