#!/usr/bin/env python3
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import platform


def target_platform():
    if FLAGS.target_platform is not None:
        return FLAGS.target_platform
    return platform.system().lower()


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
    df += """
# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libglib2.0-dev \
        libtbb-dev \
        patchelf \
        git \
        make \
        build-essential \
        wget \
        ca-certificates

# Build instructions:
# https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinux

ARG OPENVINO_VERSION
ARG OPENVINO_BUILD_TYPE
WORKDIR /workspace

# When git cloning it is important that we include '-b' and branchname
# so that this command is re-run when the branch changes, otherwise it
# will be cached by docker and continue using an old clone/branch. We
# are relying on the use of a release branch that does not change once
# it is released (if a patch is needed for that release we expect
# there to be a new version).
RUN git clone -b ${OPENVINO_VERSION} https://github.com/openvinotoolkit/openvino.git

WORKDIR /workspace/openvino
RUN git submodule update --init --recursive

WORKDIR /workspace/openvino/build
RUN /bin/bash -c 'cmake \
        -DCMAKE_BUILD_TYPE=${OPENVINO_BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=/workspace/install \
        -DENABLE_TESTS=OFF \
        -DENABLE_VALIDATION_SET=OFF \
        .. && \
    make -j$(nproc) install'

WORKDIR /opt/openvino
RUN cp -r /workspace/openvino/licensing LICENSE.openvino
RUN mkdir -p include && \
    cp -r /workspace/install/runtime/include/ngraph include/. && \
    cp -r /workspace/install/runtime/include/openvino include/.
RUN mkdir -p lib && \
    cp -P /usr/lib/x86_64-linux-gnu/libtbb.so* lib/. && \
    cp -P /workspace/install/runtime/lib/intel64/libopenvino*.so* lib/. \
"""

    df += """
RUN (cd lib && \
     for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
     done)
"""

    with open(FLAGS.output, "w") as dfile:
        dfile.write(df)


def dockerfile_for_windows(output_file):
    df = dockerfile_common()
    df += """
SHELL ["cmd", "/S", "/C"]

# Build instructions:
# https://github.com/openvinotoolkit/openvino/wiki/BuildingForWindows

ARG OPENVINO_VERSION
ARG OPENVINO_BUILD_TYPE
WORKDIR /workspace

# When git cloning it is important that we include '-b' and branchname
# so that this command is re-run when the branch changes, otherwise it
# will be cached by docker and continue using an old clone/branch. We
# are relying on the use of a release branch that does not change once
# it is released (if a patch is needed for that release we expect
# there to be a new version).
RUN git clone -b %OPENVINO_VERSION% https://github.com/openvinotoolkit/openvino.git

WORKDIR /workspace/openvino
RUN git submodule update --init --recursive

WORKDIR /workspace/openvino/build
ARG VS_DEVCMD_BAT="call \BuildTools\VC\Auxiliary\Build\vcvars64.bat"
ARG CMAKE_BAT="cmake \
          -DCMAKE_BUILD_TYPE=%OPENVINO_BUILD_TYPE% \
          -DCMAKE_INSTALL_PREFIX=C:/workspace/install \
          -DENABLE_TESTS=OFF \
          .."
ARG CMAKE_BUILD_BAT="cmake --build . --config %OPENVINO_BUILD_TYPE% --target install --verbose -j8"
RUN powershell Set-Content 'build.bat' -value '%VS_DEVCMD_BAT%','%CMAKE_BAT%','%CMAKE_BUILD_BAT%'
RUN build.bat

WORKDIR /opt/openvino
RUN xcopy /I /E \\workspace\\openvino\\licensing LICENSE.openvino
RUN mkdir include
RUN xcopy /I /E \\workspace\\install\\runtime\\include\\ie include
RUN xcopy /I /E \\workspace\\install\\runtime\\include\\ngraph include\\ngraph
RUN xcopy /I /E \\workspace\\install\\runtime\\include\\openvino include\\openvino
RUN xcopy /I /E \\workspace\\install\\runtime\\bin\\intel64\\%OPENVINO_BUILD_TYPE% bin
RUN xcopy /I /E \\workspace\\install\\runtime\\lib\\intel64\\%OPENVINO_BUILD_TYPE% lib
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\bin\\tbb12.dll bin\\tbb12.dll
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\bin\\tbb12_debug.dll bin\\tbb12_debug.dll
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\lib\\tbb.lib lib\\tbb.lib
RUN copy \\workspace\\install\\runtime\\3rdparty\\tbb\\lib\\tbb_debug.lib lib\\tbb_debug.lib
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
