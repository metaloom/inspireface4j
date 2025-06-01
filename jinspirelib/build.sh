#!/bin/bash

set -euo pipefail

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
INSPIREFACE_INCLUDE_DIR="${CURRENT_DIR}/../inspireface-linux-x86-ubuntu18-1.2.1/InspireFace/include"
INSPIREFACE_LIB_DIR="${CURRENT_DIR}/../inspireface-linux-x86-ubuntu18-1.2.1/InspireFace/lib"
#INSPIREFACE_LIB_DIR="${CURRENT_DIR}/../inspireface-linux-tensorrt-cuda12.2_ubuntu22.04-1.2.1/lib"
TENSOR_RT_DIR="${CURRENT_DIR}/../TensorRT-10.8.0.43"

# Function to display usage
usage() {
    echo "Usage: $0"
    echo
    exit 1
}

# Show usage if help is requested
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
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
    cmake .. -D INSPIREFACE_INCLUDE_DIR="${INSPIREFACE_INCLUDE_DIR}" -D INSPIREFACE_LIB_DIR="${INSPIREFACE_LIB_DIR}"  -D TENSOR_RT_DIR="${TENSOR_RT_DIR}" -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"

    echo "Building project incrementally ..."
    cmake --build . -- -j$(nproc)  # Parallel build using available CPU cores
}



build_project "Release"

echo "Build completed successfully."
