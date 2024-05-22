import shutil
import subprocess
import sys

import pytest

def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true")
    parser.addoption("--skip-download", action="store_true")
    parser.addoption("--model-cache", action="store")

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: run tests on GPU device",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords:
            if not config.getoption("--gpu"):
                item.add_marker(pytest.mark.skip("Test requires --gpu flag to be set and Intel GPU device on the host machine"))
