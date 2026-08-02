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
 * Run {@code yunet/setup-cuda.sh} once. There is nothing to source and no environment to export:
 * the script only puts files on disk, and {@link CudaNatives} — invoked from this class's static
 * initialiser, before any ONNX Runtime class is touched — configures the JVM to use them.
 *
 * <p>
 * What it is working around is that the Maven {@code onnxruntime_gpu} artifact is built against
 * <b>CUDA 12</b>, even though onnxruntime.ai documents 1.27+ as CUDA 13 — that table describes the
 * PyPI and NuGet packages, not the Java one. Verified with {@code readelf -d} on the jar's provider
 * library:
 *
 * <pre>
 *   NEEDED libcudart.so.12  libcublas.so.12  libcublasLt.so.12  libcurand.so.10
 * </pre>
 *
 * <b>cuDNN is absent from that list and is still required.</b> ONNX Runtime {@code dlopen}s
 * {@code libcudnn.so} lazily, at the first Conv node, so the provider attaches and the session is
 * created before anything complains — the failure surfaces on the first inference as
 * {@code ORT_NOT_IMPLEMENTED ... cuDNN is unavailable or disabled}. Absence from {@code NEEDED}
 * means "not linked at load time", not "not required".
 */
public final class OrtRuntime {

	private static final Logger log = LoggerFactory.getLogger(OrtRuntime.class);

	static {
		// Before anything can touch ai.onnxruntime: the native path is read once, by that
		// package's own class initialiser, and cannot be changed afterwards.
		CudaNatives.bootstrap();
	}

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
							+ "requires GPU; it will not fall back to CPU silently. Check that the "
							+ "onnxruntime_gpu artifact is on the classpath rather than "
							+ "onnxruntime, and that a CUDA runtime matching it is loadable. "
							+ cudaSetupHint() + " Pass Device.cpu() to opt out deliberately. "
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

	private static volatile Boolean cudaProbe;

	/**
	 * True if a CUDA session can actually be created in this process.
	 *
	 * <p>
	 * This attaches the provider for real rather than asking
	 * {@link OrtEnvironment#getAvailableProviders()}, which reports what the ONNX Runtime binary
	 * was <em>compiled</em> with and not what can be loaded. With the {@code onnxruntime_gpu}
	 * artifact it returns {@code [CPU, CUDA, TENSOR_RT]} on any machine, including one with no CUDA
	 * runtime installed at all — so using it as a guard passes, and the failure lands later at
	 * session creation instead.
	 *
	 * <p>
	 * The result is cached: which native libraries this process can reach is fixed by the time the
	 * first session is opened, so the answer cannot change while the JVM runs.
	 */
	public static boolean cudaAvailable() {
		Boolean cached = cudaProbe;
		if (cached == null) {
			cached = probeCuda();
			cudaProbe = cached;
		}
		return cached;
	}

	private static boolean probeCuda() {
		try {
			env();   // logger first -- see the note in open()
			try (OrtSession.SessionOptions probe = new OrtSession.SessionOptions()) {
				probe.addCUDA(0);
				return true;
			}
		} catch (Throwable t) {
			log.debug("CUDA provider will not attach: {}", t.getMessage());
			return false;
		}
	}

	/** How to fix it, quoted into errors so the remedy travels with the failure. */
	static String cudaSetupHint() {
		return "Run yunet/setup-cuda.sh once -- it downloads the ONNX Runtime build matching the "
			+ "CUDA already on this machine, plus cuDNN, into yunet/.cuda. Nothing needs to be "
			+ "sourced or exported afterwards. If this project is a dependency rather than the "
			+ "build you are running, install to ~/.facedetect4j/cuda (setup-cuda.sh --global) or "
			+ "set -Dfacedetect4j.cuda.dir=<dir>.";
	}
}
