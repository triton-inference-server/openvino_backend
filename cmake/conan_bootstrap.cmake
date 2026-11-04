# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Top-level include that makes Conan the dependency provider (cmake-conan 2.x).
# CMakeLists.txt adds it to CMAKE_PROJECT_TOP_LEVEL_INCLUDES when
# TRITON_OPENVINO_USE_CONAN is ON (the default on Linux), so a plain
# `cmake -S . -B build` is enough.
#
# Conan then runs inside the configure step, on the first find_package().
# Optional environment: CONAN_REMOTE_URL, with the credentials in the URL
# (https://<user>:<token>@host/...), to share binaries through Artifactory;
# ARTIFACTORY_USER/ARTIFACTORY_TOKEN are a fallback when the URL has none;
# when unset, only the remotes Conan already knows are used.

cmake_minimum_required(VERSION 3.31.8)

if(NOT TRITON_OPENVINO_USE_CONAN)
  message(FATAL_ERROR
    "conan_bootstrap.cmake requires -DTRITON_OPENVINO_USE_CONAN=ON")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "TRITON_OPENVINO_USE_CONAN is supported on Linux only "
    "(CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})")
endif()
if(NOT TRITON_BUILD_OPENVINO_VERSION)
  message(FATAL_ERROR "TRITON_BUILD_OPENVINO_VERSION is required")
endif()

set(_triton_conan_py "${CMAKE_CURRENT_LIST_DIR}/conan/triton_conan.py")
set(_triton_conan_graph "${CMAKE_BINARY_DIR}/conan-install-graph.json")

# The provider calls ${CONAN_COMMAND}; route it through our wrapper. The
# wrapper keeps credentials out of the logs and records the install graph.
find_program(_triton_conan_real conan)
if(NOT _triton_conan_real)
  message(FATAL_ERROR
    "TRITON_OPENVINO_USE_CONAN is ON but the conan executable was not found. "
    "Install it (pip install conan==2.33.0) or configure with "
    "-DTRITON_OPENVINO_USE_CONAN=OFF to build OpenVINO in Docker.")
endif()
set(ENV{TRITON_CONAN_EXECUTABLE} "${_triton_conan_real}")
set(ENV{TRITON_CONAN_GRAPH_FILE} "${_triton_conan_graph}")
set(CONAN_COMMAND "${_triton_conan_py}" CACHE FILEPATH "Conan command" FORCE)
set(TRITON_CONAN_GRAPH_FILE "${_triton_conan_graph}" CACHE INTERNAL "")
set(TRITON_CONAN_WRAPPER "${_triton_conan_py}" CACHE INTERNAL "")

# Make the in-tree openvino recipe available (idempotent).
execute_process(
  COMMAND "${CONAN_COMMAND}" export
          "${CMAKE_CURRENT_LIST_DIR}/../conan/openvino"
          --version "${TRITON_BUILD_OPENVINO_VERSION}"
  RESULT_VARIABLE _triton_conan_rc
  OUTPUT_QUIET
)
if(NOT _triton_conan_rc EQUAL 0)
  message(FATAL_ERROR
    "conan export of openvino/${TRITON_BUILD_OPENVINO_VERSION} failed "
    "(only versions listed in conan/openvino/conandata.yml are available)")
endif()

# Deploy every dependency package into the build tree so the backend can ship
# the OpenVINO runtime and oneTBB next to itself (see CMakeLists.txt).
set(CONAN_INSTALL_ARGS
    "--build=missing;--deployer=full_deploy;--deployer-folder=${CMAKE_BINARY_DIR}/conan-deploy"
    CACHE STRING "Command line arguments for conan install")

# Host profile: auto-detected default, then ours, then what CMake selected.
set(CONAN_HOST_PROFILE
    "default;${CMAKE_CURRENT_LIST_DIR}/../conan/profiles/linux-release;auto-cmake"
    CACHE STRING "Conan host profile")

include("${CMAKE_CURRENT_LIST_DIR}/conan/conan_provider.cmake")

# The provider above falls back to a plain find_package() when Conan does not
# supply a package, which searches /usr, /usr/local and the package registries.
# For the packages Conan owns that fallback must not pick up a system copy, so
# wrap the provider: for those packages the system search steps are switched off
# only for the duration of the lookup (find_program() and Find modules such as
# FindThreads are untouched), and a result outside the Conan output folder is
# an error. NO_DEFAULT_PATH, which covers NO_CMAKE_SYSTEM_PATH, cannot be
# passed here because the find_package() calls are made by common/core/backend.
set(TRITON_CONAN_STRICT_PACKAGES "OpenVINO;RapidJSON;Protobuf;gRPC;TBB"
    CACHE STRING "Packages that must come from Conan, never from the system")
option(TRITON_CONAN_ALLOW_SYSTEM_PACKAGES
       "Let Conan-owned packages fall back to system copies" OFF)

set(_TRITON_CONAN_SYSTEM_FIND_VARS
    CMAKE_FIND_USE_CMAKE_SYSTEM_PATH
    CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH
    CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH
    CMAKE_FIND_USE_PACKAGE_REGISTRY
    CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY)

# A macro, like conan_provide_dependency, so that the variables find_package()
# sets stay visible to the caller.
macro(triton_provide_dependency method package_name)
  set(_tcp_strict FALSE)
  if(NOT TRITON_CONAN_ALLOW_SYSTEM_PACKAGES
     AND "${package_name}" IN_LIST TRITON_CONAN_STRICT_PACKAGES)
    set(_tcp_strict TRUE)
  endif()

  if(_tcp_strict)
    foreach(_tcp_var IN LISTS _TRITON_CONAN_SYSTEM_FIND_VARS)
      if(DEFINED ${_tcp_var})
        set(_tcp_had_${_tcp_var} TRUE)
        set(_tcp_old_${_tcp_var} "${${_tcp_var}}")
      else()
        set(_tcp_had_${_tcp_var} FALSE)
      endif()
      set(${_tcp_var} OFF)
    endforeach()
  endif()

  conan_provide_dependency(${method} ${package_name} ${ARGN})

  if(_tcp_strict)
    foreach(_tcp_var IN LISTS _TRITON_CONAN_SYSTEM_FIND_VARS)
      if(_tcp_had_${_tcp_var})
        set(${_tcp_var} "${_tcp_old_${_tcp_var}}")
      else()
        unset(${_tcp_var})
      endif()
      unset(_tcp_had_${_tcp_var})
      unset(_tcp_old_${_tcp_var})
    endforeach()
    if(${package_name}_FOUND)
      string(FIND "${${package_name}_DIR}" "${CMAKE_BINARY_DIR}/conan" _tcp_pos)
      if(NOT _tcp_pos EQUAL 0)
        message(FATAL_ERROR
          "${package_name} was found at '${${package_name}_DIR}', outside the "
          "Conan output folder. Conan-owned packages must not come from the "
          "system; add it to conanfile.txt or pass "
          "-DTRITON_CONAN_ALLOW_SYSTEM_PACKAGES=ON.")
      endif()
      unset(_tcp_pos)
    endif()
  endif()
  unset(_tcp_strict)
endmacro()

cmake_language(
  SET_DEPENDENCY_PROVIDER triton_provide_dependency
  SUPPORTED_METHODS FIND_PACKAGE
)
