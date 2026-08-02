package io.metaloom.facedetect4j.yunet.onnx;

import java.nio.file.Files;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.FaceException;

/**
 * Session construction, and the one place that decides CPU vs GPU.
 *
 * <h2>CUDA is enforced, never silently substituted</h2>
 * When {@link Device#isCuda()} is requested and the CUDA execution provider cannot be attached,
 * this <b>throws</b>. It does not warn and continue. A library that quietly drops to CPU turns a
 * deployment error into a performance mystery: nothing fails, no exception is logged, and the only
 * evidence is throughput an order of magnitude low. Callers who genuinely want CPU pass
 * {@link Device#cpu()}.
 *
 * <h2>Getting the CUDA provider to load</h2>
 * The Maven {@code onnxruntime_gpu} artifact is built against <b>CUDA 12</b>, even though
 * onnxruntime.ai documents 1.27+ as CUDA 13 — that table describes the PyPI and NuGet packages,
 * not the Java one. Verified with {@code readelf -d} on the jar's provider library:
 *
 * <pre>
 *   NEEDED libcudart.so.12  libcublas.so.12  libcublasLt.so.12  libcurand.so.10
 * </pre>
 *
 * No {@code libcudnn} soname appears anywhere, so cuDNN is not required by this build. If the host
 * has only CUDA 13 (as Debian trixie does), supply CUDA 12 from pip wheels and put them on
 * {@code LD_LIBRARY_PATH} <i>before the JVM starts</i> — the dynamic loader captures its search
 * path at process start, so setting it from inside Java is too late:
 *
 * <pre>
 * python3 -m venv .venv-cuda
 * .venv-cuda/bin/pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12
 * export LD_LIBRARY_PATH="$PWD/.venv-cuda/lib/python3.13/site-packages/nvidia/cublas/lib:..."
 * </pre>
 */
public final class OrtRuntime {

	private static final Logger log = LoggerFactory.getLogger(OrtRuntime.class);

	private OrtRuntime() {
	}

	public static OrtEnvironment env() {
		return OrtEnvironment.getEnvironment();
	}

	/**
	 * Open a session on the requested device.
	 *
	 * @throws FaceException if the model is missing, or if CUDA was requested and is unavailable
	 */
	public static OrtSession open(Path model, Device device) {
		if (!Files.isRegularFile(model)) {
			throw new FaceException("model not found: " + model);
		}
		try {
			// A git-lfs pointer is ~130 bytes of text with a .onnx name. Caught here the message
			// is actionable; caught by the protobuf parser it is not.
			long size = Files.size(model);
			if (size < 10_000) {
				throw new FaceException(model + " is only " + size + " bytes -- this is almost "
					+ "certainly a git-lfs pointer file, not a model. Fetch it from "
					+ "media.githubusercontent.com/media/... rather than raw.githubusercontent.com");
			}
		} catch (java.io.IOException e) {
			throw new FaceException("cannot stat " + model, e);
		}

		// Force the environment into existence BEFORE touching SessionOptions. ONNX Runtime
		// registers its default logger as a side effect of creating the environment, and
		// addCUDA() logs during provider registration -- so calling it first fails with
		// "Attempt to use DefaultLogger but none has been registered", which reads like a
		// missing-CUDA error and is not one.
		OrtEnvironment env = env();

		OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
		try {
			if (device.isCuda()) {
				try {
					opts.addCUDA(device.ordinal());
				} catch (OrtException e) {
					throw new FaceException(
						"CUDA execution provider unavailable for " + device + ". This library "
							+ "requires GPU; it will not fall back to CPU silently. Check that "
							+ "(a) the onnxruntime_gpu artifact is on the classpath, not "
							+ "onnxruntime, and (b) a CUDA 12 runtime is on LD_LIBRARY_PATH "
							+ "before the JVM starts. Pass Device.cpu() to opt out deliberately. "
							+ "Underlying error: " + e.getMessage(), e);
				}
			} else {
				opts.setIntraOpNumThreads(
					Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
			}
			opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
			// SFace's mxnet export leaves all 174 weight tensors declared as graph inputs, which
			// makes ORT emit one "Initializer X appears in graph inputs" warning per tensor at
			// session creation. They are harmless and unfixable from here (the model would have to
			// be re-exported), so raise the floor to ERROR rather than spew 174 lines per load.
			opts.setSessionLogLevel(ai.onnxruntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
			OrtSession session = env.createSession(model.toString(), opts);
			log.debug("opened {} on {}", model.getFileName(), device);
			return session;
		} catch (OrtException e) {
			throw new FaceException("cannot open " + model + " on " + device, e);
		}
	}

	/** True if this JVM's ONNX Runtime build exposes a CUDA provider at all. */
	public static boolean cudaAvailable() {
		try {
			return OrtEnvironment.getAvailableProviders().stream()
				.anyMatch(p -> p.name().contains("CUDA"));
		} catch (Throwable t) {
			return false;
		}
	}
}
