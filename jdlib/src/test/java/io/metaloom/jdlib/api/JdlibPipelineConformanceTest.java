package io.metaloom.jdlib.api;

import static org.junit.jupiter.api.Assumptions.abort;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import io.metaloom.facedetect4j.api.AbstractFacePipelineTest;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.jdlib.Jdlib;

/**
 * The shared {@link FacePipeline} contract, run against dlib.
 *
 * <p>
 * Uses the <b>5 point</b> predictor, which is the licensing-safe one and therefore the
 * configuration that matters. Under it dlib supplies no ArcFace landmarks at all, so this run also
 * covers the contract's null-landmark path.
 */
class JdlibPipelineConformanceTest extends AbstractFacePipelineTest {

	private static final String PREDICTOR = "shape_predictor_5_face_landmarks.dat";
	private static final String EMBEDDER = "dlib_face_recognition_resnet_model_v1.dat";

	private static final List<Path> MODEL_DIRS = List.of(
		Path.of("examples"),
		Path.of("models"),
		Path.of("../../face-eval/models/dlib"));

	private static Path find(String name) {
		return MODEL_DIRS.stream().map(d -> d.resolve(name)).filter(Files::isReadable)
			.findFirst().orElse(null);
	}

	@Override
	protected void assumeBackendAvailable() {
		assumeTrue(find(PREDICTOR) != null, PREDICTOR + " not found in " + MODEL_DIRS);
		assumeTrue(find(EMBEDDER) != null, EMBEDDER + " not found in " + MODEL_DIRS);
		try {
			// Touching the native side here turns a missing libmkl_rt.so into a skip for the whole
			// class rather than an error in every test. See the module README.
			new Jdlib(find(PREDICTOR).toString());
		} catch (UnsatisfiedLinkError e) {
			abort("libjdlib.so did not load: " + e.getMessage()
				+ " -- is libmkl_rt.so on LD_LIBRARY_PATH?");
		}
	}

	@Override
	protected FacePipeline pipeline() {
		return new JdlibPipeline(
			new Jdlib(find(PREDICTOR).toString(), find(EMBEDDER).toString(), null));
	}

	/**
	 * dlib's descriptors sit in a much narrower cosine band than the ArcFace-family embedders.
	 * Measured on the shared fixtures: same identity 0.9488, different identity 0.8900 — a margin
	 * of 0.059 where YuNet+SFace gives roughly 0.48.
	 *
	 * <p>
	 * The ordering is right, so this is a property of the model rather than a fault in the
	 * adapter, but it is worth knowing before swapping backends behind this API: dlib is designed
	 * to be compared by Euclidean distance against its own 0.6 threshold, and a cosine cut-off
	 * carried over from another backend will either accept everything or reject everything.
	 */
	@Override
	protected double minimumIdentityMargin() {
		return 0.04;
	}
}
