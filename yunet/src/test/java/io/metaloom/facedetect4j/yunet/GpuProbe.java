package io.metaloom.facedetect4j.yunet;

import java.nio.file.Files;
import java.nio.file.Path;

import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;

/**
 * Whether the GPU path works end to end, used to decide skip-versus-run.
 *
 * <p>
 * Deliberately stronger than {@code OrtRuntime.cudaAvailable()}, which only attaches the provider.
 * That is not enough: ONNX Runtime loads {@code libcudnn.so} lazily at the first Conv node, so on a
 * host with the CUDA runtime but no cuDNN the provider attaches, the session opens, and the failure
 * arrives on the first inference — after every guard has already passed. This runs one real
 * detection, which is the only check that covers that.
 *
 * <p>
 * Each cheap check gates the expensive one, and the verdict is cached, so the cost is a single
 * session load per JVM rather than one per test.
 */
public final class GpuProbe {

	private static volatile Boolean usable;

	private GpuProbe() {
	}

	public static boolean usable(Path modelDir) {
		Boolean cached = usable;
		if (cached == null) {
			cached = probe(modelDir);
			usable = cached;
		}
		return cached;
	}

	private static boolean probe(Path modelDir) {
		if (!Files.isReadable(modelDir.resolve(Yunet4j.SFACE))) {
			return false;
		}
		if (!io.metaloom.facedetect4j.yunet.onnx.OrtRuntime.cudaAvailable()) {
			return false;
		}
		try (FacePipeline p = Yunet4j.pipeline(modelDir)) {
			// 64x64 is already 32-aligned, so this exercises inference rather than the padding
			// path. No face is expected; reaching the end without throwing is the whole result.
			p.detect(FaceImage.ofBgrBytes(64, 64, new byte[64 * 64 * 3]));
			return true;
		} catch (RuntimeException e) {
			System.err.println("[GpuProbe] GPU path unusable, skipping GPU tests: " + e.getMessage());
			return false;
		}
	}

	/** Why the tests were skipped, phrased so the fix is in the message. */
	public static String skipReason() {
		return "GPU path not usable here -- run yunet/setup-cuda.sh (once; nothing to source "
			+ "afterwards), or check models/ is populated";
	}
}
