import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # import inferscope from the project root
FIX = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def qwen_config():
    return json.loads((FIX / "qwen_config.json").read_text())


@pytest.fixture
def llamacpp_text():
    return (FIX / "llamacpp_metrics.txt").read_text()


@pytest.fixture
def vllm_text():
    return (FIX / "vllm_metrics.txt").read_text()


@pytest.fixture
def kernel_csv():
    return str(FIX / "kernel_trace.csv")


@pytest.fixture
def tmp_data(tmp_path):
    return tmp_path / "data"
