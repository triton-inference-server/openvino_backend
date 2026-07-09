#!/usr/bin/env python3
# Copyright 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import multiprocessing
import os
import platform


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


def usable_cpu_count():
    # Honor CPU affinity / cgroup pinning where available (Linux); fall back to
    # the logical core count on platforms without sched_getaffinity.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count() or 1


def available_memory_gb():
    # Best-effort available RAM in GiB, or None if it cannot be determined.
    # Reads MemAvailable from /proc/meminfo (Linux); returns None elsewhere so
    # callers fall back to a CPU-only parallelism default.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return None


# Memory budget the build is assumed to have for compilation. Parallelism is
# sized so the OpenVINO build's peak memory stays within this, avoiding OOM on
# swap-less builders (see TRI-1550). Unlike the ONNX Runtime build, the OpenVINO
# build runs inside a memory-limited container (`docker build --memory`, see
# TRITON_OPENVINO_DOCKER_MEMORY in CMakeLists.txt) while the generator runs on
# the host, so this budget is capped at that container limit rather than 64 GB.
MAX_BUILD_MEMORY_GB = 16.0

# Approximate RAM headroom reserved per concurrent compiler slot.
MEM_GB_PER_SLOT = 2.0


def build_parallelism():
    # Cap OpenVINO build parallelism to avoid OOM on swap-less builders
    # (see TRI-1550). Bare `-j$(nproc)` expands to the host core count, which
    # combined with memory-heavy C++ translation units exhausts memory. Each
    # concurrent compiler slot is budgeted ~MEM_GB_PER_SLOT of RAM, and the job
    # count is bounded by that budget.
    #
    # The assumed memory budget is capped at MAX_BUILD_MEMORY_GB: we never plan
    # for more than that (so a big-RAM host doesn't over-subscribe the build
    # container), nor for more than is actually available (so a small builder
    # doesn't OOM). When the available memory is unknown, we fall back to the
    # MAX_BUILD_MEMORY_GB assumption rather than to an unbounded core count.
    #
    # Computed at Dockerfile-generation time (same host that runs `docker
    # build`), matching server/build.py and onnxruntime_backend; the resulting
    # value is baked into the Dockerfile as a literal so no shell logic runs in
    # the build stage.
    cpus = usable_cpu_count()

    mem_gb = available_memory_gb()
    budget_gb = (
        min(mem_gb, MAX_BUILD_MEMORY_GB) if mem_gb is not None else MAX_BUILD_MEMORY_GB
    )

    mem_slots = max(1, int(budget_gb // MEM_GB_PER_SLOT))
    return max(1, min(cpus, mem_slots))


def dockerfile_common():
    df = """
ARG BASE_IMAGE={}
ARG OPENVINO_VERSION={}
ARG OPENVINO_BUILD_TYPE={}
""".format(
        FLAGS.triton_container, FLAGS.openvino_version, FLAGS.build_type
    )

    df += """
FROM ${BASE_IMAGE}
WORKDIR /workspace
"""

    return df


def dockerfile_for_linux(output_file):
    df = dockerfile_common()
    if os.getenv("CCACHE_REMOTE_ONLY") and os.getenv("CCACHE_REMOTE_STORAGE"):
        df += """
ENV CCACHE_REMOTE_ONLY="true" \\
    CCACHE_REMOTE_STORAGE="{}" \\
    CMAKE_CXX_COMPILER_LAUNCHER="ccache" \\
    CMAKE_C_COMPILER_LAUNCHER="ccache" \\
    CMAKE_CUDA_COMPILER_LAUNCHER="ccache" \\
    VERBOSE=1

RUN apt-get update \\
      && apt-get install -y --no-install-recommends ccache && ccache -p \\
      && rm -rf /var/lib/apt/lists/*
""".format(
            os.getenv("CCACHE_REMOTE_STORAGE")
        )

    # Cap OpenVINO build parallelism to avoid OOM on swap-less builders
    # (see TRI-1550). The value is computed in Python and baked in as a literal
    # so the build stage stays free of shell logic (portable across Debian/RHEL).
    ov_jobs = build_parallelism()
    print(
        "[INFO] OpenVINO build parallelism: -j{} (usable cores {}, "
        "MemAvailable {})".format(
            ov_jobs,
            usable_cpu_count(),
            "{:.1f} GB".format(available_memory_gb())
            if available_memory_gb() is not None
            else "unknown",
        )
    )

    df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
        ca-certificates \\
        cmake \\
        git \\
        libglib2.0-dev \\
        python3-pip \\
        wget

RUN pip3 install patchelf==0.17.2 scons

# Build instructions:
# https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinux

# The linux part is building from source, while the windows part is using
# pre-build archive.
# TODO: Unify build steps between linux and windows.

ARG OPENVINO_VERSION
ARG OPENVINO_BUILD_TYPE
WORKDIR /workspace

# When git cloning it is important that we include '-b' and branchname
# so that this command is re-run when the branch changes, otherwise it
# will be cached by docker and continue using an old clone/branch. We
# are relying on the use of a release branch that does not change once
# it is released (if a patch is needed for that release we expect
# there to be a new version).
RUN git clone --recurse-submodules -b ${OPENVINO_VERSION} https://github.com/openvinotoolkit/openvino.git

WORKDIR /workspace/openvino

RUN cmake \\
        -DCMAKE_BUILD_TYPE=${OPENVINO_BUILD_TYPE} \\
        -DCMAKE_INSTALL_PREFIX=/workspace/install \\
        -DENABLE_TESTS=OFF \\
        -DENABLE_VALIDATION_SET=OFF \\
        -S /workspace/openvino \\
        -B /workspace/openvino/build
"""

    # Parallelism is a literal (not `$(nproc)`) so the build container cannot
    # over-subscribe its RAM; see build_parallelism() and TRI-1550.
    df += """
RUN cmake --build /workspace/openvino/build -j{} -t install
""".format(
        ov_jobs
    )

    df += """
WORKDIR /opt/openvino
RUN cp -r /workspace/openvino/licensing LICENSE.openvino
RUN mkdir -p include && \\
    cp -r /workspace/install/runtime/include/* include/.
RUN mkdir -p lib && \\
    find /workspace/install/runtime/3rdparty/tbb/lib/ -name "libtbb.so*" -exec cp -v {} lib/ \\; && \\
    find /workspace/install/runtime/lib/ -name "libopenvino*.so*" -exec cp -v {} lib/ \\;
"""

    df += """
RUN (cd lib && \\
     for i in $(find . -mindepth 1 -maxdepth 1 -type f -name "*.so*"); do \\
        patchelf --set-rpath '$ORIGIN' $i; \\
     done)
"""

    df += """
CMD ["/bin/bash"]
"""

    with open(FLAGS.output, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()
    df += """
SHELL ["cmd", "/S", "/C"]

# Install instructions:
# https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-windows.html

# The windows part is using pre-build archive, while the linux part is building
# from source.
# TODO: Unify build steps between windows and linux.

ARG OPENVINO_VERSION=2025.4.1
ARG OPENVINO_BUILD_TYPE

WORKDIR /workspace
RUN IF "%OPENVINO_VERSION%"=="2023.3.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/windows/w_openvino_toolkit_windows_2023.3.0.13775.ceeafaf64f3_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2024.0.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/windows/w_openvino_toolkit_windows_2024.0.0.14509.34caeefd078_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2024.1.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/windows/w_openvino_toolkit_windows_2024.1.0.15008.f4afc983258_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2024.4.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/windows/w_openvino_toolkit_windows_2024.4.0.16579.c3152d32c9c_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2024.5.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.5/windows/w_openvino_toolkit_windows_2024.5.0.17288.7975fa5da0c_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.0.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.0/windows/openvino_toolkit_windows_2025.0.0.17942.1f68be9f594_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.1.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.1/windows/openvino_toolkit_windows_2025.1.0.18503.6fec06580ab_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.2.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/windows/openvino_toolkit_windows_2025.2.0.19140.c01cd93e24d_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.3.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.3/windows/openvino_toolkit_windows_2025.3.0.19807.44526285f24_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.4.0" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/windows/openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64.zip --output ov.zip
RUN IF "%OPENVINO_VERSION%"=="2025.4.1" curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/windows/openvino_toolkit_windows_2025.4.1.20426.82bbf0292c5_x86_64.zip --output ov.zip
RUN IF not exist ov.zip ( echo "OpenVINO version %OPENVINO_VERSION% not supported" && exit 1 )

RUN tar -xf ov.zip
RUN powershell.exe "Get-ChildItem w_openvino_toolkit_windows_* | foreach { ren $_.fullname install }"

WORKDIR /opt/openvino
RUN xcopy /I /E \\workspace\\install\\docs\\licensing LICENSE.openvino
RUN mkdir include
RUN xcopy /I /E \\workspace\\install\\runtime\\include\\* include
RUN xcopy /I /E \\workspace\\install\\runtime\\bin\\intel64\\%OPENVINO_BUILD_TYPE% bin
RUN xcopy /I /E \\workspace\\install\\runtime\\lib\\intel64\\%OPENVINO_BUILD_TYPE% lib
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\bin\\tbb12.dll bin\\tbb12.dll
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\bin\\tbb12_debug.dll bin\\tbb12_debug.dll
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\lib\\tbb12.lib lib\\tbb12.lib
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\lib\\tbb12_debug.lib lib\\tbb12_debug.lib
"""

    with open(FLAGS.output, "w") as dfile:
        dfile.write(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-container",
        type=str,
        required=True,
        help="Triton base container to use for build.",
    )
    parser.add_argument(
        "--openvino-version", type=str, required=True, help="OpenVINO version."
    )
    parser.add_argument(
        "--build-type",
        type=str,
        default="Release",
        required=False,
        help="CMAKE_BUILD_TYPE for OpenVINO build.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="File to write Dockerfile to."
    )
    parser.add_argument(
        "--target-platform",
        required=False,
        default=None,
        help='Target for build, can be "ubuntu" or "windows". If not specified, build targets the current platform.',
    )

    FLAGS = parser.parse_args()

    if target_platform() == "windows":
        dockerfile_for_windows(FLAGS.output)
    else:
        dockerfile_for_linux(FLAGS.output)
