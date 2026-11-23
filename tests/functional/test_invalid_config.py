# Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
import common
import pytest
from model_config import INVALID_CONFIG_MODELS
from tritonclient.grpc import service_pb2


@pytest.mark.parametrize("model", list(INVALID_CONFIG_MODELS))
class TestInvalidConfig:
    repository_models = list(INVALID_CONFIG_MODELS)

    def test_model_not_ready(self, model, triton_server):
        grpc_stub = common.prepare_grpc_stub(triton_server)
        request = service_pb2.ModelReadyRequest(name=model)
        response = grpc_stub.ModelReady(request)
        assert not response.ready, f"Model {model} with an invalid config is ready"

    def test_load_error_names_parameter(self, model, triton_server):
        grpc_stub = common.prepare_grpc_stub(triton_server)
        response = grpc_stub.RepositoryIndex(service_pb2.RepositoryIndexRequest())
        entry = next(m for m in response.models if m.name == model)
        expected = INVALID_CONFIG_MODELS[model]["expected_error"]
        assert (
            expected in entry.reason
        ), f"Expected a load error naming {expected}, got: {entry.reason!r}"
