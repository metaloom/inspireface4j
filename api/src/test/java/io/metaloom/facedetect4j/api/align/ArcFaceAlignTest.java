package io.metaloom.facedetect4j.api.align;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.BoundingBox;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.Landmarks;

/**
 * Alignment is the quietest failure mode in the whole library: a wrong template, permuted
 * keypoints or a mirrored transform all produce a crop that still looks like a face and still
 * embeds without error. Nothing downstream reports a problem — the embeddings are merely worse. So
 * this is checked against an external reference rather than against a second copy of the same
 * maths.
 */
class ArcFaceAlignTest {

	/**
	 * Golden 2x3 affines from {@code skimage.transform.SimilarityTransform.estimate()} — the exact
	 * function insightface calls. Generated in float64; the tolerance below is what a correct
	 * implementation actually achieves, not a number relaxed until the test passed.
	 */
	@Test
	@DisplayName("umeyama matches skimage SimilarityTransform to 1e-9")
	void matchesSkimage() {
		assertAffine("frontal",
			new double[][] { { 247.9, 258.6 }, { 366.4, 255.1 }, { 309.2, 325.4 },
				{ 257.8, 383.9 }, { 356.1, 381.0 } },
			new double[] { 0.3119715871305828, -0.0017194978916056727, -39.34724868728452,
				0.0017194978916056727, 0.3119715871305828, -28.70841636320189 });

		// Rolled head: the rotation terms are large here, so a transposed V would flip their sign.
		assertAffine("rolled",
			new double[][] { { 210.0, 300.0 }, { 320.5, 260.4 }, { 280.1, 340.9 },
				{ 250.7, 410.2 }, { 345.0, 372.6 } },
			new double[] { 0.308336802936647, -0.09921707400692423, 2.7216456730508938,
				0.09921707400692421, 0.308336802936647, -59.85901620030893 });

		// A tiny face, upscaled: exercises scale > 1 and sub-pixel keypoints.
		assertAffine("small",
			new double[][] { { 10.25, 12.5 }, { 28.75, 12.0 }, { 19.5, 20.125 },
				{ 12.0, 28.5 }, { 27.0, 28.25 } },
			new double[] { 2.208356047441731, -0.010651717004551035, 13.179180637153515,
				0.010651717004551033, 2.2083560474417316, 26.91865265653015 });

		// Eyes and mouth corners swapped in x. Umeyama's determinant branch must handle the
		// reflection rather than emitting a mirroring transform.
		assertAffine("mirrored",
			new double[][] { { 366.4, 255.1 }, { 247.9, 258.6 }, { 309.2, 325.4 },
				{ 356.1, 381.0 }, { 257.8, 383.9 } },
			new double[] { 0.057341786789052995, 0.004388277651135274, 36.98694792761779,
				-0.004388277651135274, 0.057341786789052995, 54.85484241024287 });
	}

	private static void assertAffine(String name, double[][] src, double[] expected) {
		double[] m = ArcFaceAlign.umeyama(src, ArcFaceAlign.template());
		for (int i = 0; i < 6; i++) {
			assertThat(m[i]).as("%s m[%d]", name, i).isCloseTo(expected[i], within(1e-9));
		}
	}

	@Test
	@DisplayName("recovers a known similarity transform exactly")
	void recoversAKnownTransform() {
		// Self-validating and oracle-free: build the source by applying a known similarity to the
		// template, then the estimate must be its exact inverse. Any bug that is merely
		// self-consistent still fails this, because the answer is fixed by construction.
		double angle = 0.37, scale = 2.5, tx = 120.0, ty = -45.0;
		double a = scale * Math.cos(angle), b = -scale * Math.sin(angle);
		double c = scale * Math.sin(angle), d = scale * Math.cos(angle);

		double[][] tpl = ArcFaceAlign.template();
		double[][] src = new double[5][2];
		for (int i = 0; i < 5; i++) {
			src[i][0] = a * tpl[i][0] + b * tpl[i][1] + tx;
			src[i][1] = c * tpl[i][0] + d * tpl[i][1] + ty;
		}

		double[] m = ArcFaceAlign.umeyama(src, tpl);

		// Round-trip: mapping the source points through the estimate must land on the template.
		for (int i = 0; i < 5; i++) {
			double x = m[0] * src[i][0] + m[1] * src[i][1] + m[2];
			double y = m[3] * src[i][0] + m[4] * src[i][1] + m[5];
			assertThat(x).as("point %d x", i).isCloseTo(tpl[i][0], within(1e-9));
			assertThat(y).as("point %d y", i).isCloseTo(tpl[i][1], within(1e-9));
		}
	}

	@Test
	@DisplayName("the template maps to itself as the identity")
	void templateIsAFixedPoint() {
		double[] m = ArcFaceAlign.umeyama(ArcFaceAlign.template(), ArcFaceAlign.template());
		assertThat(m[0]).isCloseTo(1.0, within(1e-12));
		assertThat(m[1]).isCloseTo(0.0, within(1e-12));
		assertThat(m[2]).isCloseTo(0.0, within(1e-9));
		assertThat(m[3]).isCloseTo(0.0, within(1e-12));
		assertThat(m[4]).isCloseTo(1.0, within(1e-12));
		assertThat(m[5]).isCloseTo(0.0, within(1e-9));
	}

	@Test
	@DisplayName("the transform never mirrors, even on reflected input")
	void neverProducesAReflection() {
		// A negative determinant means the crop is mirrored. It would still embed cleanly and just
		// score worse, so nothing else in the stack would notice.
		double[][] reflected = { { 366.4, 255.1 }, { 247.9, 258.6 }, { 309.2, 325.4 },
			{ 356.1, 381.0 }, { 257.8, 383.9 } };
		double[] m = ArcFaceAlign.umeyama(reflected, ArcFaceAlign.template());
		assertThat(m[0] * m[4] - m[1] * m[3]).as("determinant").isGreaterThan(0);
	}

	@Test
	@DisplayName("the template is the insightface arcface_dst constants, and is defensively copied")
	void templateIsCanonicalAndImmutable() {
		double[][] t = ArcFaceAlign.template();
		assertThat(t[0]).containsExactly(38.2946, 51.6963);
		assertThat(t[4]).containsExactly(70.7299, 92.2041);

		// Callers get a copy: mutating it must not poison every later alignment in the JVM.
		t[0][0] = -999;
		assertThat(ArcFaceAlign.template()[0][0]).isEqualTo(38.2946);
	}

	@Test
	@DisplayName("produces a 112x112 crop that actually sampled the image")
	void alignProducesACrop() {
		FaceImage img = gradient(400, 400);
		Face face = faceWithLandmarks(new float[] {
			150, 170, 250, 170, 200, 220, 160, 270, 240, 270 });

		AlignedFace crop = ArcFaceAlign.align(img, face);

		assertThat(crop.bgr()).hasSize(AlignedFace.SIZE * AlignedFace.SIZE * 3);
		assertThat(crop.affine2x3()).hasSize(6);
		// A warp that sampled entirely outside the image returns all zeros and is otherwise
		// indistinguishable from a working one.
		assertThat(crop.stddev()).as("crop variance").isGreaterThan(1.0);
		assertThat(crop.source()).isSameAs(face);
	}

	@Test
	@DisplayName("landmarks far outside the image give a blank crop rather than an exception")
	void warpOutsideTheImageIsBlank() {
		FaceImage img = gradient(64, 64);
		Face face = faceWithLandmarks(new float[] {
			5000, 5000, 5100, 5000, 5050, 5050, 5010, 5100, 5090, 5100 });

		AlignedFace crop = ArcFaceAlign.align(img, face);
		assertThat(crop.stddev()).isEqualTo(0.0);
	}

	@Test
	@DisplayName("aligning a face with no landmarks fails loudly")
	void alignWithoutLandmarksThrows() {
		Face face = Face.of(new BoundingBox(0, 0, 10, 10), 0.9f, null);
		assertThatThrownBy(() -> ArcFaceAlign.align(gradient(32, 32), face))
			.isInstanceOf(FaceException.class)
			.hasMessageContaining("without landmarks");
	}

	private static Face faceWithLandmarks(float[] xy) {
		return Face.of(new BoundingBox(140, 150, 260, 290), 0.99f, new Landmarks(xy));
	}

	/** Deterministic non-uniform image, so a crop that sampled anything has non-zero variance. */
	private static FaceImage gradient(int w, int h) {
		byte[] bgr = new byte[w * h * 3];
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int o = (y * w + x) * 3;
				bgr[o] = (byte) (x % 256);
				bgr[o + 1] = (byte) (y % 256);
				bgr[o + 2] = (byte) ((x + y) % 256);
			}
		}
		return FaceImage.ofBgrBytes(w, h, bgr);
	}
}
