package io.metaloom.facedetect4j.api;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.align.ArcFaceAlign;

/**
 * The behaviour {@link FacePipeline} supplies for free, checked against a fake backend.
 *
 * <p>
 * The fake is itself part of the point: if this file compiles, the API is implementable with no
 * model, no native library and no GPU, which is what makes it a usable seam rather than a
 * description of one particular backend.
 */
class FacePipelineTest {

	@Test
	@DisplayName("primaryFace prefers a central face over a slightly larger one at the edge")
	void primaryFacePrefersTheCentre() {
		// "Largest" alone picks the bystander here. The centre weighting is the entire reason this
		// helper exists, so it is checked with a case where the two rules disagree.
		FaceImage img = blank(1000, 1000);
		Face centre = at(450, 450, 100, 100);
		Face edgeButBigger = at(0, 0, 130, 130);

		Face picked = new FakePipeline().primaryFace(img, List.of(edgeButBigger, centre))
			.orElseThrow();
		assertThat(picked).isSameAs(centre);
	}

	@Test
	@DisplayName("primaryFace still prefers size when the sizes differ enough")
	void primaryFacePrefersSizeWhenItDominates() {
		FaceImage img = blank(1000, 1000);
		Face tinyCentre = at(490, 490, 20, 20);
		Face largeOffCentre = at(600, 600, 300, 300);

		Face picked = new FakePipeline().primaryFace(img, List.of(tinyCentre, largeOffCentre))
			.orElseThrow();
		assertThat(picked).isSameAs(largeOffCentre);
	}

	@Test
	@DisplayName("primaryFace picks the subject, not a centred bystander, on a 4K frame")
	void primaryFaceIsResolutionIndependent() {
		// The case insightface's absolute `area - 2*dist^2` gets wrong: at 3840x2160 a 400x400
		// face 700 px off centre scores -820000 there and loses to a 60x60 face at the centre.
		FaceImage frame = blank(3840, 2160);
		Face subject = at(1020, 880, 400, 400);
		Face centredBystander = at(1890, 1050, 60, 60);

		Face picked = new FakePipeline().primaryFace(frame, List.of(centredBystander, subject))
			.orElseThrow();
		assertThat(picked).isSameAs(subject);
	}

	@Test
	@DisplayName("the same geometry gives the same winner at every resolution")
	void scalingTheImageDoesNotChangeTheChoice() {
		// The property the old formula lacked. Identical relative layout, four resolutions: if the
		// answer ever differs, the scoring has an absolute term in it again.
		for (int scale : new int[] { 1, 2, 4, 8 }) {
			int w = 480 * scale;
			int h = 270 * scale;
			Face bystander = at(0.02f * w, 0.02f * h, 0.10f * w, 0.10f * h);
			Face subject = at(0.55f * w, 0.50f * h, 0.18f * w, 0.18f * h);

			Face picked = new FakePipeline()
				.primaryFace(blank(w, h), List.of(bystander, subject)).orElseThrow();
			assertThat(picked).as("at %dx%d", w, h).isSameAs(subject);
		}
	}

	@Test
	@DisplayName("a corner face must be more than twice the area of a central one to win")
	void cornerDiscountIsBounded() {
		// The discount is bounded, unlike a penalty that grows without limit: the biggest face
		// still wins if it is big enough, which is what stops the rule collapsing into "most
		// central". CENTER_PULL = 0.5 is exactly the 2x boundary, so this pins the constant.
		FaceImage img = blank(1000, 1000);
		Face centre = at(450, 450, 100, 100);   // area 10000

		Face justUnder = centredAt(1000, 1000, 140);   // 19600 * 0.5 = 9800
		assertThat(new FakePipeline().primaryFace(img, List.of(justUnder, centre)).orElseThrow())
			.as("1.96x at the corner loses").isSameAs(centre);

		Face justOver = centredAt(1000, 1000, 145);    // 21025 * 0.5 = 10512
		assertThat(new FakePipeline().primaryFace(img, List.of(justOver, centre)).orElseThrow())
			.as("2.10x at the corner wins").isSameAs(justOver);
	}

	@Test
	@DisplayName("of two equally sized faces the more central one always wins")
	void scoreFallsOffMonotonically() {
		FaceImage img = blank(1000, 1000);
		FakePipeline p = new FakePipeline();
		// primaryFace exposes no score, so monotonicity is observed through the choice.
		for (int offset = 100; offset <= 400; offset += 100) {
			Face nearer = centredAt(500, 500 - (offset - 100), 100);
			Face further = centredAt(500, 500 - offset, 100);
			assertThat(p.primaryFace(img, List.of(further, nearer)).orElseThrow())
				.as("offset %d loses to offset %d", offset, offset - 100).isSameAs(nearer);
		}
	}

	@Test
	@DisplayName("primaryFace on no faces is empty rather than an exception")
	void primaryFaceOfNothing() {
		assertThat(new FakePipeline().primaryFace(blank(10, 10), List.of())).isEmpty();
	}

	@Test
	@DisplayName("detectAndEmbed keeps landmark-less faces, without an embedding")
	void detectAndEmbedKeepsUnembeddableFaces() {
		// Dropping them would lose a detection the caller may still want the box for; embedding
		// them would mean aligning from nothing.
		FakePipeline p = new FakePipeline();
		p.faces.add(withLandmarks(at(10, 10, 50, 50)));
		p.faces.add(at(100, 100, 50, 50));   // no landmarks

		List<Face> out = p.detectAndEmbed(blank(200, 200));

		assertThat(out).hasSize(2);
		assertThat(out.get(0).hasEmbedding()).isTrue();
		assertThat(out.get(1).hasEmbedding()).isFalse();
		assertThat(out.get(1).box()).isEqualTo(p.faces.get(1).box());
	}

	@Test
	@DisplayName("detectAndEmbed embeds every face exactly once")
	void detectAndEmbedDoesNotDoubleWork() {
		FakePipeline p = new FakePipeline();
		p.faces.add(withLandmarks(at(10, 10, 50, 50)));
		p.faces.add(withLandmarks(at(80, 80, 50, 50)));

		p.detectAndEmbed(blank(200, 200));
		assertThat(p.embedCalls).isEqualTo(2);
	}

	@Test
	@DisplayName("a FacePipeline is usable as either half of itself")
	void isBothADetectorAndAnEmbedder() {
		// Callers should be able to hold the narrow interface they need. If this stops compiling,
		// the hierarchy has been broken.
		FacePipeline p = new FakePipeline();
		FaceDetector detector = p;
		FaceEmbedder embedder = p;
		assertThat(detector.device()).isEqualTo(embedder.device());
	}

	private static FaceImage blank(int w, int h) {
		return FaceImage.ofBgrBytes(w, h, new byte[w * h * 3]);
	}

	private static Face at(float x, float y, float w, float h) {
		return Face.of(BoundingBox.ofXYWH(x, y, w, h), 0.9f, null);
	}

	/** A square face whose <em>centre</em> sits at (cx, cy) — what the scoring actually uses. */
	private static Face centredAt(float cx, float cy, float size) {
		return at(cx - size / 2, cy - size / 2, size, size);
	}

	private static Face withLandmarks(Face f) {
		BoundingBox b = f.box();
		float[] xy = {
			b.x1() + b.width() * 0.3f, b.y1() + b.height() * 0.4f,
			b.x1() + b.width() * 0.7f, b.y1() + b.height() * 0.4f,
			b.centerX(), b.centerY(),
			b.x1() + b.width() * 0.35f, b.y1() + b.height() * 0.75f,
			b.x1() + b.width() * 0.65f, b.y1() + b.height() * 0.75f
		};
		return Face.of(b, f.score(), new Landmarks(xy));
	}

	/** A backend with no model behind it. Proves the API carries no hidden runtime assumptions. */
	private static final class FakePipeline implements FacePipeline {

		final List<Face> faces = new ArrayList<>();
		int embedCalls;

		@Override
		public List<Face> detect(FaceImage image) {
			return List.copyOf(faces);
		}

		@Override
		public float[] embed(FaceImage image, Face face) {
			embedCalls++;
			return embed(align(image, face));
		}

		@Override
		public float[] embed(AlignedFace aligned) {
			float[] v = new float[4];
			v[0] = 1f;   // unit length, as the contract requires
			return v;
		}

		@Override
		public AlignedFace align(FaceImage image, Face face) {
			return ArcFaceAlign.align(image, face);
		}

		@Override
		public int dimensions() {
			return 4;
		}

		@Override
		public Device device() {
			return Device.cpu();
		}

		@Override
		public void close() {
		}
	}
}
