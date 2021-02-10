#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--triton-container',
                        type=str,
                        required=True,
                        help='Triton base container to use for build.')
    parser.add_argument('--openvino-version',
                        type=str,
                        required=True,
                        help='OpenVINO version.')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='File to write Dockerfile to.')

    FLAGS = parser.parse_args()

    df = '''
ARG BASE_IMAGE={}
ARG OPENVINO_VERSION={}
'''.format(FLAGS.triton_container, FLAGS.openvino_version)

    df += '''
FROM ${BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        patchelf

WORKDIR /workspace

ARG OPENVINO_VERSION
ENV INTEL_OPENVINO_DIR /opt/intel/openvino_${OPENVINO_VERSION}
ENV LD_LIBRARY_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64:$INTEL_OPENVINO_DIR/deployment_tools/ngraph/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:/usr/local/openblas/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH $INTEL_OPENVINO_DIR/tools:$PYTHONPATH
ENV IE_PLUGINS_PATH $INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64

RUN wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021 && rm GPG-PUB-KEY-INTEL-OPENVINO-2021 && \
    cd /etc/apt/sources.list.d && \
    echo "deb https://apt.repos.intel.com/openvino/2021 all main">intel-openvino-2021.list && \
    apt update && \
    apt install -y intel-openvino-dev-ubuntu20-${OPENVINO_VERSION}
# && \
#    cd ${INTEL_OPENVINO_DIR}/install_dependencies && ./install_openvino_dependencies.sh

ARG INTEL_COMPUTE_RUNTIME_URL=https://github.com/intel/compute-runtime/releases/download/19.41.14441
RUN wget ${INTEL_COMPUTE_RUNTIME_URL}/intel-gmmlib_19.3.2_amd64.deb && \
    wget ${INTEL_COMPUTE_RUNTIME_URL}/intel-igc-core_1.0.2597_amd64.deb && \
    wget ${INTEL_COMPUTE_RUNTIME_URL}/intel-igc-opencl_1.0.2597_amd64.deb && \
    wget ${INTEL_COMPUTE_RUNTIME_URL}/intel-opencl_19.41.14441_amd64.deb && \
    wget ${INTEL_COMPUTE_RUNTIME_URL}/intel-ocloc_19.41.14441_amd64.deb && \
    dpkg -i *.deb && rm -rf *.deb

#
# Copy all artifacts needed by the backend
#
WORKDIR /opt/openvino

RUN mkdir -p /opt/openvino/include && \
   (cd /opt/openvino/include && \
     cp -r /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/include/* \
          .)

RUN mkdir -p /opt/openvino/lib && \
    cp -r /opt/intel/openvino_${OPENVINO_VERSION}/licensing \
          /opt/openvino/LICENSE.openvino && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_legacy.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_transformations.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/ngraph/lib/libngraph.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/external/tbb/lib/libtbb.so.2 \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/external/tbb/lib/libtbbmalloc.so.2 \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libMKLDNNPlugin.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_lp_transformations.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_ir_reader.so \
       /opt/openvino/lib && \
    cp /opt/intel/openvino_${OPENVINO_VERSION}/deployment_tools/inference_engine/lib/intel64/libinference_engine_onnx_reader.so \
       /opt/openvino/lib && \
    (cd /opt/openvino/lib && \
     chmod a-x * && \
     ln -sf libtbb.so.2 libtbb.so && \
     ln -sf libtbbmalloc.so.2 libtbbmalloc.so)
'''

    df += '''
RUN cd /opt/openvino/lib && \
    for i in `find . -mindepth 1 -maxdepth 1 -type f -name '*\.so*'`; do \
        patchelf --set-rpath '$ORIGIN' $i; \
    done
'''

    with open(FLAGS.output, "w") as dfile:
        dfile.write(df)
