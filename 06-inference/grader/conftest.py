"""Instructor-owned fixtures; the public runner always selects toyvllm."""

import importlib
from types import SimpleNamespace
import pytest
import torch
from grader.oracle.model import TinyConfig as OracleConfig, make_model as oracle_model


def pytest_addoption(parser):
    parser.addoption("--implementation", default="toyvllm", help="Instructor adapter namespace")
    parser.addoption("--case-seed", default=1729, type=int, help="Reproducible extended-case seed")


def pytest_configure(config):
    torch.set_num_threads(1)
    torch.manual_seed(config.getoption("--case-seed"))


@pytest.fixture
def case_seed(request):
    return request.config.getoption("--case-seed")


@pytest.fixture
def api(request):
    base = request.config.getoption("--implementation")
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"{base}.{name}")
            for name in [
                "tokenizer",
                "model",
                "sampling",
                "cache",
                "engine",
                "server",
                "metrics",
                "artifacts",
            ]
        }
    )


@pytest.fixture
def cfg(api):
    return api.model.TinyConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64)


@pytest.fixture
def oracle(cfg):
    return oracle_model(seed=101, cfg=OracleConfig(**vars(cfg)))


@pytest.fixture
def model(api, cfg, oracle):
    result = api.model.make_model(seed=17, cfg=cfg)
    result.load_state_dict(oracle.state_dict(), strict=True)
    return result.eval()
