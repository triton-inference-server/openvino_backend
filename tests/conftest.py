import shutil
import subprocess
import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: run tests on GPU device",
    )


def pytest_runtest_setup(item):
    for mark in item.iter_markers():
        if "gpu" in mark.name:
            if sys.platform.startswith("linux"):
                process = subprocess.run(
                    ["/bin/bash", "-c", 'lspci | grep -E "VGA|3D"'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    shell=False,
                )
                if process.returncode != 0:
                    pytest.skip("Test requires Intel GPU device on the host machine")
            elif sys.platform.startswith("win") and "win" not in item.config.getoption(
                "--image_os"
            ):
                wsl = shutil.which("wsl")
                if not wsl:
                    pytest.skip(
                        "Test requires Intel GPU device and configured WSL2 on the host machine"
                    )
