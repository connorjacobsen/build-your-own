import importlib
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--implementation", default="learner.api")
    parser.addoption("--case-seed", type=int, default=1729)


def pytest_configure(config):
    torch.set_num_threads(1)


@pytest.fixture
def api(request):
    return importlib.import_module(request.config.getoption("--implementation"))


@pytest.fixture(params=[0, 1, 2], ids=["case-a", "case-b", "case-c"])
def seed(request):
    return request.config.getoption("--case-seed") + request.param * 101
