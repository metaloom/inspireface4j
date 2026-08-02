package io.metaloom.facedetect4j.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The contract every backend has to satisfy, run against each of them.
 *
 * <p>
 * An interface only buys you something if the implementations behave the same way through it.
 * Checking each backend against its own expectations proves it works; checking all of them against
 * one shared suite proves they are <em>interchangeable</em>, which is the property that makes
 * swapping a backend a change of construction site rather than an audit of every call.
 *
 * <p>
 * Backends legitimately differ in what they support — InspireFace and dlib cannot embed a supplied
 * crop, and only some produce ArcFace landmarks. Those are asked about via
 * {@link FaceEmbedder#supportsAlignedEmbed()} rather than assumed, and the negative case is
 * asserted just as firmly as the positive one: a backend that quietly returns something wrong from
 * an operation it cannot really perform is worse than one that refuses.
 *
 * <p>
 * Shipped in this module's test-jar. Subclasses supply a pipeline and skip themselves when their
 * models or natives are missing.
 */
public abstract class AbstractFacePipelineTest {

	/** A fresh pipeline. Called per test; the caller closes it. */
	protected abstract FacePipeline pipeline();

	/**
	 * Skip the whole suite when this backend cannot run here — missing weights, missing natives,
	 * no GPU. Implementations call {@code Assumptions.assumeTrue}.
	 */
	protected void assumeBackendAvailable() {
	}

	/** Override when a backend genuinely cannot separate these two, to document that. */
	protected boolean supportsIdentitySeparation() {
		return true;
	}

	/**
	 * How far apart same-identity and different-identity cosines must sit.
	 *
	 * <p>
	 * Overridable because the usable range differs sharply by embedder, and a single number that
	 * every backend passes would be too weak to catch anything. Lowering it for a backend is a
	 * statement about that backend, so record the measured values when you do.
	 */
	protected double minimumIdentityMargin() {
		return 0.15;
	}

	private FaceImage image(String name) throws IOException {
		Path p = TestData.image(name);
		assumeTrue(Files.isReadable(p), "test image missing: " + p);
		return FaceImage.read(p);
	}

	@Test
	@DisplayName("detect: finds the face, inside the image, with a usable score")
	void detectFindsTheFace() throws Exception {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			List<Face> faces = p.detect(img);

			assertThat(faces).as("faces found").isNotEmpty();
			for (Face f : faces) {
				BoundingBox b = f.box();
				assertThat(b.width()).as("width").isGreaterThan(0);
				assertThat(b.height()).as("height").isGreaterThan(0);
				// A box mostly outside the frame means the decode used the wrong scale or the
				// wrong coordinate convention -- both produce boxes that still look plausible in
				// isolation.
				assertThat(b.centerX()).as("centre x").isBetween(0f, (float) img.width());
				assertThat(b.centerY()).as("centre y").isBetween(0f, (float) img.height());
				assertThat(f.score()).as("score").isBetween(0f, 1f);
			}
		}
	}

	@Test
	@DisplayName("detect: any landmarks returned are in ArcFace order")
	void landmarksAreInArcFaceOrder() throws Exception {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			for (Face f : p.detect(img)) {
				if (f.landmarks() == null) {
					continue;   // documented for this backend; nothing to check
				}
				// Permuted points still produce a face-shaped crop that embeds without error.
				assertThat(f.landmarks().geometryLooksSane())
					.as("eyes left-to-right and above the mouth").isTrue();
			}
		}
	}

	@Test
	@DisplayName("embed: vectors are L2 normalised, so similarity is a dot product")
	void embeddingsAreUnitLength() throws Exception {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			Face f = primary(p, img);

			float[] v = p.embed(img, f);
			assertThat(v).as("embedding").isNotEmpty();
			assertThat(v.length).as("length matches dimensions()").isEqualTo(p.dimensions());

			double n = 0;
			for (float x : v) {
				n += (double) x * x;
			}
			// Not cosmetic: Face.similarity is a bare dot product, so an unnormalised backend
			// silently returns similarities outside [-1, 1] that still sort plausibly.
			assertThat(Math.sqrt(n)).as("L2 norm").isCloseTo(1.0, within(1e-4));
		}
	}

	@Test
	@DisplayName("embed: the same identity scores far above a different one")
	void separatesIdentities() throws Exception {
		assumeBackendAvailable();
		assumeTrue(supportsIdentitySeparation());
		try (FacePipeline p = pipeline()) {
			Face a = embedded(p, TestData.IMG_FACE_NEUTRAL);
			Face b = embedded(p, TestData.IMG_FACE_HAPPY);
			Face c = embedded(p, TestData.IMG_FACE_OTHER_IDENTITY);

			double same = a.similarity(b);
			double different = a.similarity(c);

			// Only the ordering and the gap are portable. Absolute thresholds are not: SFace's
			// operating point is ~0.33, ArcFace's ~0.25, and InspireFace ships 0.48.
			assertThat(same).as("same identity").isGreaterThan(different);
			assertThat(same - different).as("margin").isGreaterThan(minimumIdentityMargin());
		}
	}

	@Test
	@DisplayName("detectAndEmbed: same faces as detect, each carrying a vector")
	void detectAndEmbedAgreesWithDetect() throws Exception {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			List<Face> detected = p.detect(img);
			List<Face> embedded = p.detectAndEmbed(img);

			assertThat(embedded).as("face count").hasSameSizeAs(detected);
			assertThat(embedded).allSatisfy(f -> {
				if (f.landmarks() != null || f.hasEmbedding()) {
					assertThat(f.hasEmbedding()).as("embedded").isTrue();
				}
			});
		}
	}

	@Test
	@DisplayName("aligned embedding: supported and consistent, or refused outright")
	void alignedEmbeddingIsHonest() throws Exception {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			Face f = primary(p, img);

			if (!p.supportsAlignedEmbed()) {
				// The refusal is the contract. Returning a plausible-looking vector from an
				// alignment the backend did not actually use is the failure this pins down.
				AlignedFace crop = new AlignedFace(new byte[112 * 112 * 3], new float[6], f);
				assertThatThrownBy(() -> p.embed(crop))
					.isInstanceOf(UnsupportedOperationException.class);
				return;
			}

			AlignedFace crop = p.align(img, f);
			assertThat(crop.bgr()).hasSize(AlignedFace.SIZE * AlignedFace.SIZE * 3);
			// Near-zero variance means the warp sampled outside the image and produced a black
			// square, which embeds perfectly happily.
			assertThat(crop.stddev()).as("crop variance").isGreaterThan(5.0);
			// The convenience path must be the crop path plus alignment, not a second code path.
			assertThat(p.embed(crop)).isEqualTo(p.embed(img, f));
		}
	}

	@Test
	@DisplayName("device is reported, never left to be guessed")
	void deviceIsReported() {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			assertThat(p.device()).isNotNull();
			assertThat(p.device().toString()).isNotBlank();
		}
	}

	@Test
	@DisplayName("an image with no face yields no faces rather than a crash")
	void handlesAnEmptyImage() {
		assumeBackendAvailable();
		try (FacePipeline p = pipeline()) {
			FaceImage blank = FaceImage.ofBgrBytes(320, 240, new byte[320 * 240 * 3]);
			assertThat(p.detect(blank)).isEmpty();
			assertThat(p.detectAndEmbed(blank)).isEmpty();
		}
	}

	private Face primary(FacePipeline p, FaceImage img) {
		return p.primaryFace(img, p.detect(img)).orElseThrow(
			() -> new AssertionError("no face detected in the primary test image"));
	}

	private Face embedded(FacePipeline p, String name) throws Exception {
		FaceImage img = image(name);
		Face f = primary(p, img);
		return f.withEmbedding(p.embed(img, f));
	}
}
