#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
INSPIREFACE_VERSION="${INSPIREFACE_VERSION:-1.2.3}"
INSPIREFACE_DIR="${CURRENT_DIR}/../inspireface-linux-x86-ubuntu18-${INSPIREFACE_VERSION}/InspireFace"
INSPIREFACE_INCLUDE_DIR="${INSPIREFACE_DIR}/include"
INSPIREFACE_LIB_DIR="${INSPIREFACE_DIR}/lib"
TENSOR_RT_DIR="${CURRENT_DIR}/../TensorRT-10.8.0.43"

# OpenCV 5 build (or install) directory that contains OpenCVConfig.cmake.
# Override via the OpenCV_DIR environment variable or the first argument.
OPENCV_DIR="${1:-${OpenCV_DIR:-${CURRENT_DIR}/../../opencv/build}}"

RESOURCES_DIR="${CURRENT_DIR}/../src/main/resources/native/linux"

# Function to display usage
usage() {
    echo "Usage: $0 [OpenCV_DIR]"
    echo
    echo "  OpenCV_DIR  Directory containing OpenCVConfig.cmake of an OpenCV 5 build."
    echo "              Defaults to \$OpenCV_DIR or ../../opencv/build"
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

# Bundle the matching InspireFace runtime next to the built binding
echo "Copying libInspireFace.so (${INSPIREFACE_VERSION}) to ${RESOURCES_DIR} ..."
mkdir -p "${RESOURCES_DIR}"
cp "${INSPIREFACE_LIB_DIR}/libInspireFace.so" "${RESOURCES_DIR}/libInspireFace.so"

# The 1.2.3 release ships with an executable stack which the JVM refuses to load
python3 "${CURRENT_DIR}/clear-execstack.py" "${RESOURCES_DIR}/libInspireFace.so"

echo "Build completed successfully."
