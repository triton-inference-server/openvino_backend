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

cmake_minimum_required(VERSION 3.31.8)

#
# Choose where OpenVINO comes from. This must happen before project(): the
# Conan dependency provider can only be installed through
# CMAKE_PROJECT_TOP_LEVEL_INCLUDES, which project() consumes.
#
# TRITON_OPENVINO_USE_CONAN (default ON on Linux, the only platform the Conan
# build supports) gets OpenVINO from Conan. Set it OFF to build OpenVINO in
# Docker instead, which requires TRITON_BUILD_CONTAINER.
#
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(_triton_openvino_use_conan_default ON)
else()
  set(_triton_openvino_use_conan_default OFF)
endif()
option(TRITON_OPENVINO_USE_CONAN "Get OpenVINO from Conan instead of Docker"
       ${_triton_openvino_use_conan_default})
if(TRITON_OPENVINO_USE_CONAN)
  if(NOT TRITON_BUILD_OPENVINO_VERSION)
    # Keep in sync with conanfile.txt and conan/openvino/conandata.yml.
    set(TRITON_BUILD_OPENVINO_VERSION "2026.4.1")
  endif()
  list(APPEND CMAKE_PROJECT_TOP_LEVEL_INCLUDES
       "${CMAKE_CURRENT_LIST_DIR}/cmake/conan_bootstrap.cmake")
endif()

project(tritonopenvinobackend LANGUAGES C CXX)
message(STATUS "TRITON_OPENVINO_USE_CONAN=${TRITON_OPENVINO_USE_CONAN}")

# Use C++20 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 20 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

#
# Set TRITON_BUILD_OPENVINO_VERSION to the version of OpenVINO that
# you want to be built for the backend.
#
# Set TRITON_BUILD_CONTAINER to the container to use as a base for the
# build. Required on all platforms: the image is never inferred here, so
# that whoever drives the build decides it.
#
# TRITON_OPENVINO_USE_CONAN (see the top of this file) selects
# Conan instead of Docker for OpenVINO and the other third-party dependencies.
# Conan runs during configure through the CMake dependency provider, see
# cmake/conan_bootstrap.cmake. TRITON_BUILD_CONTAINER is not used in that mode.
#
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
set(TRITON_BUILD_CONTAINER "" CACHE STRING "Container to use as a base for build")
set(TRITON_BUILD_OPENVINO_VERSION "" CACHE STRING "OpenVINO version to build")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")

set(TRITON_ENABLE_GPU OFF CACHE BOOL "GPU disabled" FORCE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT TRITON_BUILD_OPENVINO_VERSION)
  message(FATAL_ERROR "TRITON_BUILD_OPENVINO_VERSION is required")
endif()

if(NOT TRITON_OPENVINO_USE_CONAN AND NOT TRITON_BUILD_CONTAINER)
  message(FATAL_ERROR "TRITON_BUILD_CONTAINER is required")
endif()

if(TRITON_OPENVINO_USE_CONAN AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "TRITON_OPENVINO_USE_CONAN is supported on Linux only")
endif()

set(TRITON_OPENVINO_INCLUDE_PATHS "${CMAKE_CURRENT_BINARY_DIR}/openvino/include")
set(TRITON_OPENVINO_LIB_PATHS "${CMAKE_CURRENT_BINARY_DIR}/openvino/lib")
set(TRITON_OPENVINO_DOCKER_IMAGE "tritonserver_openvino")
set(TRITON_OPENVINO_DOCKER_MEMORY "16g")
if (WIN32)
  set(OPENVINO_LIBRARY "openvino.lib")
else()
  set(OPENVINO_LIBRARY "openvino.so")
endif() # WIN32

#
# Conan-provided dependencies
#
if(TRITON_OPENVINO_USE_CONAN)
  find_package(OpenVINO CONFIG REQUIRED)
  find_package(RapidJSON CONFIG REQUIRED)
  # common/ reads RAPIDJSON_INCLUDE_DIRS; Conan's CMakeDeps names it
  # RapidJSON_INCLUDE_DIRS.
  if(NOT RAPIDJSON_INCLUDE_DIRS)
    set(RAPIDJSON_INCLUDE_DIRS "${RapidJSON_INCLUDE_DIRS}")
  endif()
endif()

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_openvino.ldscript libtriton_openvino.ldscript COPYONLY)

add_library(
  triton-openvino-backend SHARED
  src/openvino.cc
  src/openvino_utils.cc
  src/openvino_utils.h
)

add_library(
  TritonOpenVINOBackend::triton-openvino-backend ALIAS triton-openvino-backend
)

target_include_directories(
  triton-openvino-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(NOT TRITON_OPENVINO_USE_CONAN)
  target_include_directories(
    triton-openvino-backend
    PRIVATE
      ${TRITON_OPENVINO_INCLUDE_PATHS}
  )
endif()

target_compile_features(triton-openvino-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-openvino-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Wno-ignored-qualifiers -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

set_target_properties(
  triton-openvino-backend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_openvino
    SKIP_BUILD_RPATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "$\{ORIGIN\}"
)

if(NOT WIN32)
  set_target_properties(
    triton-openvino-backend
    PROPERTIES
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_openvino.ldscript
      LINK_FLAGS "-Wl,--version-script libtriton_openvino.ldscript"
  )
endif() # NOT WIN32

target_link_libraries(
  triton-openvino-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

if(TRITON_OPENVINO_USE_CONAN)
  target_link_libraries(
    triton-openvino-backend
    PRIVATE
      openvino::runtime     # from Conan
  )
else()
  FOREACH(p ${TRITON_OPENVINO_LIB_PATHS})
    target_link_directories(
      triton-openvino-backend
      PRIVATE ${p}
    )
  ENDFOREACH(p)

  target_link_libraries(
    triton-openvino-backend
    PRIVATE
      ${OPENVINO_LIBRARY}
  )
endif() # TRITON_OPENVINO_USE_CONAN

#
# Build the OpenVINO libraries using docker.
#
if(TRITON_OPENVINO_USE_CONAN)
  # OpenVINO comes from Conan; nothing to build here.
elseif (WIN32)
  add_custom_command(
    OUTPUT
      openvino/lib/${OPENVINO_LIBRARY}
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_openvino_dockerfile.py --build-type="${CMAKE_BUILD_TYPE}" --triton-container="${TRITON_BUILD_CONTAINER}" --openvino-version="${TRITON_BUILD_OPENVINO_VERSION}" --output=Dockerfile.openvino
    COMMAND docker build --memory ${TRITON_OPENVINO_DOCKER_MEMORY} --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE} --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE}_cache1 -t ${TRITON_OPENVINO_DOCKER_IMAGE} -f ./Dockerfile.openvino ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND powershell.exe -noprofile -c "docker rm openvino_backend_ov > $null 2>&1; if ($LASTEXITCODE) { 'error ignored...' }; exit 0"
    COMMAND docker create --name openvino_backend_ov ${TRITON_OPENVINO_DOCKER_IMAGE}
    COMMAND rmdir /s/q openvino
    COMMAND docker cp openvino_backend_ov:/opt/openvino openvino
    COMMAND docker rm openvino_backend_ov
    COMMENT "Building OpenVino"
  )
else()
  add_custom_command(
    OUTPUT
      openvino/lib/${OPENVINO_LIBRARY}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_openvino_dockerfile.py --build-type="${CMAKE_BUILD_TYPE}" --triton-container="${TRITON_BUILD_CONTAINER}" --openvino-version="${TRITON_BUILD_OPENVINO_VERSION}" --output=Dockerfile.openvino
    COMMAND docker build --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE} --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE}_cache0 --cache-from=${TRITON_OPENVINO_DOCKER_IMAGE}_cache1 -t ${TRITON_OPENVINO_DOCKER_IMAGE} -f ./Dockerfile.openvino ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND docker rm openvino_backend_ov || echo "error ignored..." || true
    COMMAND docker create --name openvino_backend_ov ${TRITON_OPENVINO_DOCKER_IMAGE}
    COMMAND rm -fr openvino
    COMMAND docker cp openvino_backend_ov:/opt/openvino openvino
    COMMAND docker rm openvino_backend_ov
    COMMENT "Building OpenVino"
  )
endif() # WIN32

if(NOT TRITON_OPENVINO_USE_CONAN)
  add_custom_target(ov_target DEPENDS openvino/lib/${OPENVINO_LIBRARY})
  add_library(openvino-library SHARED IMPORTED GLOBAL)
  add_dependencies(openvino-library ov_target)
  add_dependencies(triton-openvino-backend openvino-library)
  set_target_properties(
    openvino-library
    PROPERTIES
      IMPORTED_LOCATION openvino/lib/${OPENVINO_LIBRARY}
  )
endif() # NOT TRITON_OPENVINO_USE_CONAN

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOpenVINOBackend)

if(NOT TRITON_OPENVINO_USE_CONAN)
  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/openvino/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
    PATTERN *lib EXCLUDE
    PATTERN *bin EXCLUDE
    PATTERN *include EXCLUDE
  )
endif()

if (NOT WIN32)
  install(
    TARGETS
      triton-openvino-backend
    EXPORT
      triton-openvino-backend-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
  )

  if(TRITON_OPENVINO_USE_CONAN)
    # The Conan install step deploys every dependency package under
    # conan-deploy/ (see cmake/conan_bootstrap.cmake). Ship the OpenVINO
    # runtime, its plugins and oneTBB next to the backend, as the Docker flow
    # does, and point their RUNPATH at $ORIGIN so they find each other.
    file(
      GLOB_RECURSE TRITON_OPENVINO_RUNTIME_LIBS
      "${CMAKE_BINARY_DIR}/conan-deploy/full_deploy/host/openvino/*/lib/lib*.so*"
      "${CMAKE_BINARY_DIR}/conan-deploy/full_deploy/host/onetbb/*/lib/libtbb*.so*"
    )
    if(NOT TRITON_OPENVINO_RUNTIME_LIBS)
      message(FATAL_ERROR
        "No OpenVINO runtime libraries found under "
        "${CMAKE_BINARY_DIR}/conan-deploy; the Conan deployer output layout "
        "may have changed")
    endif()
    install(
      FILES ${TRITON_OPENVINO_RUNTIME_LIBS}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
    )
    find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
    install(CODE "set(TRITON_PATCHELF \"${PATCHELF_EXECUTABLE}\")")
    install(CODE "set(TRITON_BACKEND_DIR \"${CMAKE_INSTALL_PREFIX}/backends/openvino\")")
    install(CODE [[
      file(GLOB _libs "$ENV{DESTDIR}${TRITON_BACKEND_DIR}/*.so*")
      foreach(_lib ${_libs})
        execute_process(
          COMMAND ${TRITON_PATCHELF} --set-rpath "$ORIGIN" ${_lib}
          COMMAND_ERROR_IS_FATAL ANY
        )
      endforeach()
    ]])
  else()
    install(
      DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/openvino/lib/
      USE_SOURCE_PERMISSIONS
      DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
    )
  endif() # TRITON_OPENVINO_USE_CONAN
else()
  install(
    TARGETS
      triton-openvino-backend
    EXPORT
      triton-openvino-backend-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
  )

  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/openvino/bin/
    USE_SOURCE_PERMISSIONS
    DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/openvino
  )
endif() # WIN32

install(
  EXPORT
    triton-openvino-backend-targets
  FILE
    TritonOpenVINOBackendTargets.cmake
  NAMESPACE
    TritonOpenVINOBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOpenVINOBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenVINOBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenVINOBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-openvino-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenVINOBackendTargets.cmake
  NAMESPACE TritonOpenVINOBackend::
)

export(PACKAGE TritonOpenVINOBackend)

#
# Push the Conan packages that were built from source during configure.
# Opt in with: cmake --build build --target conan-upload
# Needs CONAN_REMOTE_URL in the environment (see cmake/conan_bootstrap.cmake);
# packages restored from a cache or a remote are never uploaded.
#
if(TRITON_OPENVINO_USE_CONAN)
  set(TRITON_CONAN_UPLOAD_REMOTE "" CACHE STRING
      "Conan remote for the conan-upload target (default: CONAN_REMOTE_URL)")
  option(TRITON_CONAN_UPLOAD_DRY_RUN "Do not upload, only report" OFF)
  set(_triton_conan_upload_args --graph "${TRITON_CONAN_GRAPH_FILE}")
  if(TRITON_CONAN_UPLOAD_REMOTE)
    list(APPEND _triton_conan_upload_args --remote "${TRITON_CONAN_UPLOAD_REMOTE}")
  endif()
  if(TRITON_CONAN_UPLOAD_DRY_RUN)
    list(APPEND _triton_conan_upload_args --dry-run)
  endif()
  add_custom_target(
    conan-upload
    COMMAND ${CMAKE_COMMAND} -E env
            "TRITON_CONAN_EXECUTABLE=${_triton_conan_real}"
            "${TRITON_CONAN_WRAPPER}" triton-upload ${_triton_conan_upload_args}
    COMMENT "Uploading Conan packages built from source"
    VERBATIM
  )
endif()
