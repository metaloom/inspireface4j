package io.metaloom.jdlib;

import static org.junit.jupiter.api.Assumptions.abort;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.TestData;
import io.metaloom.jdlib.util.FaceDescriptor;

/**
 * Proves that the bundled {@code libjdlib.so} actually loads and that dlib returns something.
 *
 * <p>
 * Worth having because {@link Jdlib}'s loader catches {@code Exception}, which does not cover
 * {@link UnsatisfiedLinkError}: a missing dependency of the {@code .so} throws out of the
 * constructor, while a missing {@code .so} resource is swallowed and only surfaces later, on the
 * first native call, in whatever code happened to make it. Both are covered here.
 *
 * <p>
 * Skipped rather than failed when the models or the MKL runtime are absent, so the module still
 * builds on a machine that has neither.
 */
class JdlibSmokeTest {

	/**
	 * The 5 point predictor on purpose: the 68 point one is not licensed for commercial use, and a
	 * test that quietly depends on it invites everything downstream to do the same.
	 */
	private static final String PREDICTOR = "shape_predictor_5_face_landmarks.dat";

	private static final Path IMAGE = TestData.image(TestData.IMG_FACE_NEUTRAL);

	/** Wherever the models happen to live; none of these are committed. */
	private static final List<Path> MODEL_DIRS = List.of(
		Path.of("examples"),
		Path.of("models"),
		Path.of("../../face-eval/models/dlib"));

	private static Path predictor() {
		return MODEL_DIRS.stream()
			.map(d -> d.resolve(PREDICTOR))
			.filter(Files::isReadable)
			.findFirst()
			.orElse(null);
	}

	@Test
	@DisplayName("the native library loads and HOG detection finds a face")
	void detectsAFace() throws Exception {
		Path model = predictor();
		assumeTrue(model != null, PREDICTOR + " not found in " + MODEL_DIRS
			+ " -- fetch it from http://dlib.net/files/");
		assumeTrue(Files.isReadable(IMAGE), "test image missing: " + IMAGE);

		BufferedImage img = ImageIO.read(new File(IMAGE.toString()));

		Jdlib jdlib;
		List<Rectangle> faces;
		try {
			jdlib = new Jdlib(model.toString());
			faces = jdlib.detectFace(img);
		} catch (UnsatisfiedLinkError e) {
			// libjdlib.so needs libmkl_rt.so, which is not part of any base install. See the README.
			abort("libjdlib.so did not load: " + e.getMessage()
				+ " -- is libmkl_rt.so on LD_LIBRARY_PATH?");
			return;
		}

		assumeTrue(!faces.isEmpty(), "dlib found no face -- HOG is scale sensitive, not a build problem");
		Rectangle box = faces.get(0);
		if (box.width <= 0 || box.height <= 0) {
			throw new AssertionError("degenerate box from the native layer: " + box);
		}

		// The landmark path exercises a second handler and the JNI list marshalling, which is where
		// a struct-layout mismatch between the .so and the Java class would show up.
		List<FaceDescriptor> landmarks = jdlib.getFaceLandmarks(img);
		if (landmarks.isEmpty() || landmarks.get(0).getFacialLandmarks().size() != 5) {
			throw new AssertionError("expected 5 landmarks per face, got " + landmarks);
		}
	}
}
