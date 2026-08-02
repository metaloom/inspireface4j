#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
INSPIREFACE_VERSION="${INSPIREFACE_VERSION:-1.2.3}"
INSPIREFACE_DIR="${CURRENT_DIR}/../inspireface-linux-x86-ubuntu18-${INSPIREFACE_VERSION}/InspireFace"
INSPIREFACE_INCLUDE_DIR="${INSPIREFACE_DIR}/include"
INSPIREFACE_LIB_DIR="${INSPIREFACE_DIR}/lib"
TENSOR_RT_DIR="${CURRENT_DIR}/../TensorRT-10.8.0.43"

# OpenCV build (or install) directory that contains OpenCVConfig.cmake. This has to be the same
# OpenCV major that opencv-ffm binds -- see the rationale in CMakeLists.txt, which enforces it.
# Override via the OpenCV_DIR environment variable or the first argument.
OPENCV_DIR="${1:-${OpenCV_DIR:-${CURRENT_DIR}/../../../opencv-4.10.0/build}}"

RESOURCES_DIR="${CURRENT_DIR}/../src/main/resources/native/linux"

# Function to display usage
usage() {
    echo "Usage: $0 [OpenCV_DIR]"
    echo
    echo "  OpenCV_DIR  Directory containing OpenCVConfig.cmake, of the same OpenCV major that"
    echo "              opencv-ffm is built against (currently 4.10)."
    echo "              Defaults to \$OpenCV_DIR or ../../../opencv-4.10.0/build"
    echo
    echo "Environment:"
    echo "  INSPIREFACE_VERSION  InspireFace release to build against (default: 1.2.3)"
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

if [[ ! -d "${INSPIREFACE_INCLUDE_DIR}" ]]; then
    echo "ERROR: InspireFace ${INSPIREFACE_VERSION} not found at ${INSPIREFACE_DIR}"
    echo "Download and extract inspireface-linux-x86-ubuntu18-${INSPIREFACE_VERSION}.zip first."
    exit 1
fi

if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
    echo "ERROR: No OpenCVConfig.cmake in ${OPENCV_DIR}"
    echo "Point this script at an OpenCV 5 build directory (see usage)."
    exit 1
fi

# Detect platform and architecture
platform=$(uname -s)
architecture=$(uname -m)

# Function to build the project
build_project() {
    local build_type="${1:-Release}"
    local build_dir="${CURRENT_DIR}/build"

    # Ensure the build directory exists
    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "Configuring CMake with build type: $build_type ..."
    cmake .. -D INSPIREFACE_INCLUDE_DIR="${INSPIREFACE_INCLUDE_DIR}" -D INSPIREFACE_LIB_DIR="${INSPIREFACE_LIB_DIR}" -D OpenCV_DIR="${OPENCV_DIR}" -D TENSOR_RT_DIR="${TENSOR_RT_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building project incrementally ..."
    cmake --build . -- -j$(nproc)  # Parallel build using available CPU cores
}



build_project "Release"

# CMake links libjinspireface.so straight into ${RESOURCES_DIR}, so the build is already done.
#
# libInspireFace.so is deliberately NOT copied from the release here. The 1.2.3 release ships with
# an executable stack, which the JVM refuses to load; the copy in resources has already had that
# bit cleared and is the one that works. Overwriting it with the raw release file - which is what
# this script used to do, followed by a clear-execstack.py that is not in the tree - leaves a
# runtime that fails at load with no indication that a build step did it.
if [[ ! -f "${RESOURCES_DIR}/libInspireFace.so" ]]; then
    echo "ERROR: ${RESOURCES_DIR}/libInspireFace.so is missing." >&2
    echo "Restore it from git rather than copying ${INSPIREFACE_LIB_DIR}/libInspireFace.so:" >&2
    echo "  the release file needs its PT_GNU_STACK executable bit cleared first." >&2
    exit 1
fi

echo "Build completed successfully -- wrote ${RESOURCES_DIR}/libjinspireface.so"
