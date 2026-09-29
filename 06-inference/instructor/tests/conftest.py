import torch
import pytest
from grader.oracle import TinyConfig, make_model


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture
def model():
    return make_model(cfg=TinyConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64))
