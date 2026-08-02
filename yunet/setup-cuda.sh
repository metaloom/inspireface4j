#!/bin/bash
# Make the GPU path work, using whichever CUDA the machine already has.
#
#   ./setup-cuda.sh
#   mvn test
#
# That is the whole procedure. There is no environment to source and no variable to export: this
# script only puts files on disk, and io.metaloom.facedetect4j.yunet.onnx.CudaNatives finds and
# loads them from inside the JVM. See that class for why LD_LIBRARY_PATH is not needed.
#
# Use --global to install into ~/.facedetect4j/cuda instead of ./.cuda, which is what an
# application that merely depends on this library wants -- the search from the working directory
# upward will not reach into this checkout from another project's build.
#
# ---------------------------------------------------------------------------------------------
# THE PROBLEM
#
# ONNX Runtime 1.28 ships two different native builds of the same version:
#
#   Maven  com.microsoft.onnxruntime:onnxruntime_gpu:1.28.0  ->  CUDA 12
#   PyPI   onnxruntime-gpu 1.28.0                            ->  CUDA 13
#
#   $ readelf -d libonnxruntime_providers_cuda.so | grep -o 'libcud.*so\.[0-9]*'
#     Maven jar : libcudart.so.12  libcublas.so.12  libcublasLt.so.12  libcurand.so.10
#     PyPI wheel: libcudart.so.13  libcublas.so.13  libcublasLt.so.13
#
# onnxruntime.ai's "1.27+ is CUDA 13" table describes the PyPI and NuGet packages, not the Java
# one. So on a CUDA 13 host the Maven jar cannot load its provider at all, and the error names no
# library:
#
#   OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library
#
# and with a CUDA 12 that is present but too old (12.6, say) it gets further, then fails with:
#
#   libonnxruntime_providers_cuda.so: undefined symbol: cudaLibraryGetKernel, version libcudart.so.12
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS SCRIPT DOES
#
# Detects the CUDA on the machine and adapts, rather than insisting on one:
#
#   CUDA 13 present  ->  fetch the CUDA-13 ORT natives from the PyPI wheel (~280 MB) and point the
#                        JVM at them. Uses the system CUDA; nothing is duplicated.
#   CUDA 12 present  ->  use it as-is with the Maven build, download nothing.
#   Neither          ->  fetch a project-local CUDA 12 from NVIDIA's redistributable tarballs.
#
# Either way cuDNN 9 is fetched if missing, because ORT needs it and no distro ships it. It does
# NOT appear in the NEEDED list above, because ORT dlopen()s libcudnn.so lazily at the first Conv
# node -- absence from NEEDED means "not linked at load time", not "not required". The session
# opens fine and inference then fails with
#
#   ORT_NOT_IMPLEMENTED ... cuDNN is unavailable or disabled for CUDA Execution Provider
#
# Everything downloaded lands under one directory and nothing outside it is touched. Delete that
# directory to undo. No Python, no pip, no venv, no sudo.
#
# CUDA itself can of course come from NVIDIA's packages instead -- the debs drop an
# /etc/ld.so.conf.d entry, so after `ldconfig` those libraries are on the default loader path and
# this script will find and use them. cuDNN cannot: NVIDIA publishes no Debian repository for it
# (compute/cudnn/repos has no debian tree, and the debian13 CUDA repo carries zero cuDNN
# packages), so the tarball below is the only route on this distribution. For CUDA 12 specifically
# you need the *debian12* repo; the debian13 one starts at CUDA 13.1.
#
# Set FORCE_CUDA12=1 to ignore a system CUDA 13 and use the stock Maven build instead.
# ---------------------------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

# ldconfig lives in /sbin, which is not on a normal user's PATH on Debian. Without it the system
# CUDA is invisible here and the script would download a second copy of what is already installed.
export PATH="$PATH:/sbin:/usr/sbin"

# Pinned so a build is reproducible. Keep ORT_VERSION equal to the ort.version property in the
# parent pom -- the Java JNI shim and the core library have to come from the same release.
ORT_VERSION=1.28.0
CUDA_RELEASE=12.9.1      # only used when there is no system CUDA at all
CUDNN_RELEASE=9.25.0

CUDA_BASE=https://developer.download.nvidia.com/compute/cuda/redist
CUDNN_BASE=https://developer.download.nvidia.com/compute/cudnn/redist
DEST=.cuda
FORCE12=${FORCE_CUDA12:-}

case "${1:-}" in
	--global) DEST="$HOME/.facedetect4j/cuda" ;;
	"") ;;
	*) echo "usage: $0 [--global]" >&2; exit 2 ;;
esac

for tool in curl tar; do
	command -v "$tool" > /dev/null || { echo "need $tool on PATH" >&2; exit 1; }
done

# Absolute from here on, so --global and the default behave identically everywhere below.
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)

# --------------------------------------------------------------------------------------------
# Locating libraries
# --------------------------------------------------------------------------------------------

# Echoes the directory holding $1, or LDCONFIG when it is already on the default loader path.
find_lib_dir() {
	local soname=$1 dir
	if ldconfig -p 2> /dev/null | grep -q "$soname"; then
		echo LDCONFIG
		return 0
	fi
	for dir in "$DEST"/*/lib "$DEST"/* /usr/local/cuda-1[23]*/targets/x86_64-linux/lib \
		/usr/local/cuda-1[23]*/lib64 /usr/lib/x86_64-linux-gnu; do
		[ -e "$dir/$soname" ] && { echo "$dir"; return 0; }
	done
	return 1
}

# --------------------------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------------------------

download() {
	local url=$1 out=$2
	echo "==> downloading $(basename "$out")"
	# Progress bar only on a terminal: curl's meter redraws with carriage returns, which in a CI
	# log or a piped build becomes several hundred lines of noise per file.
	local progress=(--no-progress-meter)
	[ -t 1 ] && progress=(--progress-bar)
	curl -fSL --retry 3 "${progress[@]}" -o "$out.part" "$url"
	mv "$out.part" "$out"
}

resolve() {
	local manifest=$1 component=$2 variant=${3:-}
	command -v jq > /dev/null || { echo "need jq to read NVIDIA's release manifest" >&2; exit 1; }
	if [ -n "$variant" ]; then
		jq -r --arg c "$component" --arg v "$variant" \
			'.[$c]["linux-x86_64"][$v] | "\(.relative_path) \(.sha256)"' <<< "$manifest"
	else
		jq -r --arg c "$component" \
			'.[$c]["linux-x86_64"] | "\(.relative_path) \(.sha256)"' <<< "$manifest"
	fi
}

# Fetch an NVIDIA redistributable and unpack only its shared libraries. The tarballs also carry
# static archives and headers, which together are roughly two thirds of the unpacked size and are
# useless to a JVM that only ever dlopen()s.
fetch_nvidia() {
	local base=$1 rel_path=$2 sha=$3
	local archive="$DEST/$(basename "$rel_path")"
	[ -f "$archive" ] || download "$base/$rel_path" "$archive"

	# Verify before unpacking: a truncated download unpacks far enough to look plausible and then
	# fails at dlopen with a message that says nothing about the download.
	if [ -n "$sha" ] && [ "$sha" != null ] && command -v sha256sum > /dev/null; then
		local actual
		actual=$(sha256sum "$archive" | cut -d' ' -f1)
		if [ "$actual" != "$sha" ]; then
			echo "checksum mismatch for $(basename "$archive")" >&2
			echo "  expected $sha" >&2
			echo "  actual   $actual" >&2
			rm -f "$archive"
			exit 1
		fi
	fi

	echo "    unpacking $(basename "$archive")"
	tar -xf "$archive" -C "$DEST" --exclude='*.a' --exclude='*/include/*'
	rm -f "$archive"
}

fetch_cudnn() {
	local cuda_major=$1 manifest path sha
	echo "==> fetching cuDNN $CUDNN_RELEASE for CUDA $cuda_major"
	manifest=$(curl -fsSL "$CUDNN_BASE/redistrib_$CUDNN_RELEASE.json")
	read -r path sha <<< "$(resolve "$manifest" cudnn "cuda$cuda_major")"
	fetch_nvidia "$CUDNN_BASE" "$path" "$sha"
}

# The CUDA 13 ORT natives, taken from the PyPI wheel because Maven publishes only the CUDA 12
# build. The wheel carries no Java JNI shim, so that one library still comes from the Maven jar:
# its SONAME requirement (libonnxruntime.so.1) is satisfied by the wheel's core library of the
# same ORT version, which is what makes the mix work. Same version on both sides is essential.
#
# Sets ORT_NATIVE_PATH rather than echoing it: this function also prints progress, and capturing
# its stdout would splice a "==> downloading" line into the middle of the path.
ORT_NATIVE_PATH=""
fetch_ort_cuda13() {
	local dir="$DEST/ort-cuda13"
	ORT_NATIVE_PATH="$dir"
	if [ -f "$dir/libonnxruntime_providers_cuda.so" ] && [ -f "$dir/libonnxruntime4j_jni.so" ]; then
		echo "==> CUDA 13 ORT natives already present"
		return 0
	fi
	command -v unzip > /dev/null || { echo "need unzip to open the wheel" >&2; exit 1; }
	mkdir -p "$dir"

	local whl="$DEST/onnxruntime_gpu.whl" url
	url=$(curl -fsSL https://pypi.org/simple/onnxruntime-gpu/ \
		| grep -oE "https://[^\"#]*onnxruntime_gpu-$ORT_VERSION-cp313-cp313-manylinux[^\"#]*x86_64\.whl" \
		| head -1)
	[ -n "$url" ] || { echo "no ONNX Runtime $ORT_VERSION linux wheel on PyPI" >&2; exit 1; }
	[ -f "$whl" ] || download "$url" "$whl"

	unzip -qo "$whl" 'onnxruntime/capi/libonnxruntime*.so*' -d "$DEST/whl"
	cp "$DEST/whl/onnxruntime/capi/libonnxruntime.so.$ORT_VERSION" "$dir/libonnxruntime.so.1"
	ln -sf libonnxruntime.so.1 "$dir/libonnxruntime.so"
	cp "$DEST/whl/onnxruntime/capi/libonnxruntime_providers_shared.so" "$dir/"
	cp "$DEST/whl/onnxruntime/capi/libonnxruntime_providers_cuda.so" "$dir/"
	rm -rf "$DEST/whl" "$whl"

	local jar="$HOME/.m2/repository/com/microsoft/onnxruntime/onnxruntime_gpu/$ORT_VERSION/onnxruntime_gpu-$ORT_VERSION.jar"
	if [ ! -f "$jar" ]; then
		echo "not in ~/.m2 yet: $jar" >&2
		echo "run 'mvn -pl yunet dependency:resolve' from the reactor root first" >&2
		exit 1
	fi
	unzip -qo "$jar" 'ai/onnxruntime/native/linux-x64/libonnxruntime4j_jni.so' -d "$DEST/jar"
	cp "$DEST/jar/ai/onnxruntime/native/linux-x64/libonnxruntime4j_jni.so" "$dir/"
	rm -rf "$DEST/jar"
}

# --------------------------------------------------------------------------------------------
# Decide
# --------------------------------------------------------------------------------------------

native_path=""

cudart12=$(find_lib_dir libcudart.so.12 || true)
cudart13=$(find_lib_dir libcudart.so.13 || true)

if [ -n "$cudart13" ] && [ -z "$FORCE12" ]; then
	echo "==> system CUDA 13 found -- using it, with ONNX Runtime's CUDA 13 build"
	echo "    libcudart.so.13 : $cudart13"
	fetch_ort_cuda13
	native_path=$ORT_NATIVE_PATH

elif [ -n "$cudart12" ]; then
	echo "==> system CUDA 12 found -- using it, with the Maven onnxruntime_gpu build"
	echo "    libcudart.so.12 : $cudart12"

else
	echo "==> no system CUDA found -- fetching a project-local CUDA $CUDA_RELEASE"
	manifest=$(curl -fsSL "$CUDA_BASE/redistrib_$CUDA_RELEASE.json")
	for component in cuda_cudart libcublas libcurand cuda_nvrtc; do
		read -r path sha <<< "$(resolve "$manifest" "$component")"
		fetch_nvidia "$CUDA_BASE" "$path" "$sha"
	done
fi

# cuDNN, whichever branch we took. Matched to the CUDA major in use.
cuda_major=12
[ -n "$native_path" ] && cuda_major=13
cudnn=$(find_lib_dir libcudnn.so.9 || true)
if [ -z "$cudnn" ]; then
	fetch_cudnn "$cuda_major"
	cudnn=$(find_lib_dir libcudnn.so.9 || true)
	[ -n "$cudnn" ] || { echo "cuDNN still not found after fetching -- aborting" >&2; exit 1; }
fi
echo "    libcudnn.so.9   : $cudnn"

# An env file from an older version of this script is worse than no file: sourcing it puts a
# JAVA_TOOL_OPTIONS on the environment that overrides what the JVM now configures for itself, and
# it may point at natives this run has just replaced.
if [ -f .cuda-env ]; then
	rm -f .cuda-env
	echo "==> removed .cuda-env (no longer used -- nothing needs sourcing)"
fi

# Report rather than assume: the libraries are only half of it, and a green setup on a box with no
# driver is exactly the false confidence this script exists to avoid.
if command -v nvidia-smi > /dev/null 2>&1; then
	echo "==> GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
	echo "==> WARNING: no nvidia-smi. The libraries are in place but there may be no driver." >&2
fi

echo
echo "Done -- $DEST"
if [ -n "$native_path" ]; then
	echo "ONNX Runtime will load its natives from $native_path"
fi
echo "Nothing to source. Just run: mvn test"
