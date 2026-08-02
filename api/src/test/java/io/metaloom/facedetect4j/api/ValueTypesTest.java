package io.metaloom.facedetect4j.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.awt.image.BufferedImage;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/** The value types every backend has to agree on. Pure JDK — no models, no natives, no GPU. */
class ValueTypesTest {

	@Nested
	class Boxes {

		@Test
		@DisplayName("xywh and corner forms describe the same box")
		void xywh() {
			BoundingBox b = BoundingBox.ofXYWH(10, 20, 30, 40);
			assertThat(b).isEqualTo(new BoundingBox(10, 20, 40, 60));
			assertThat(b.width()).isEqualTo(30f);
			assertThat(b.height()).isEqualTo(40f);
			assertThat(b.area()).isEqualTo(1200f);
			assertThat(b.centerX()).isEqualTo(25f);
			assertThat(b.centerY()).isEqualTo(40f);
		}

		@Test
		@DisplayName("an inverted box has zero area rather than a negative one")
		void invertedBoxHasNoArea() {
			// Negative area would make primaryFace's largest-face comparison rank a degenerate
			// detection above every real one.
			assertThat(new BoundingBox(50, 50, 10, 10).area()).isEqualTo(0f);
		}

		@Test
		@DisplayName("iou: identical is 1, disjoint is 0, half-overlap is 1/3")
		void iou() {
			BoundingBox a = new BoundingBox(0, 0, 10, 10);
			assertThat(a.iou(a)).isCloseTo(1.0, within(1e-9));
			assertThat(a.iou(new BoundingBox(100, 100, 110, 110))).isEqualTo(0.0);
			// Overlap 50, union 150.
			assertThat(a.iou(new BoundingBox(5, 0, 15, 10))).isCloseTo(50.0 / 150.0, within(1e-9));
		}

		@Test
		@DisplayName("boxes that merely touch do not overlap")
		void touchingIsNotOverlapping() {
			// The edge case that decides whether NMS merges two adjacent faces into one.
			BoundingBox a = new BoundingBox(0, 0, 10, 10);
			assertThat(a.iou(new BoundingBox(10, 0, 20, 10))).isEqualTo(0.0);
		}

		@Test
		@DisplayName("scaling maps back from letterboxed coordinates")
		void scaled() {
			assertThat(new BoundingBox(10, 20, 30, 40).scaled(2f))
				.isEqualTo(new BoundingBox(20, 40, 60, 80));
		}
	}

	@Nested
	class LandmarkPoints {

		private static final float[] SANE = { 30, 50, 70, 50, 50, 65, 35, 85, 65, 85 };

		@Test
		@DisplayName("indexed access follows ArcFace point order")
		void accessors() {
			Landmarks lm = new Landmarks(SANE);
			assertThat(lm.x(0)).isEqualTo(30f);
			assertThat(lm.y(0)).isEqualTo(50f);
			assertThat(lm.x(4)).isEqualTo(65f);
			assertThat(lm.y(4)).isEqualTo(85f);
		}

		@Test
		@DisplayName("anything but five points is rejected at construction")
		void wrongPointCountThrows() {
			// 68-point dlib output and 106-point InspireFace output both arrive here eventually;
			// failing at construction beats producing a silently wrong crop.
			assertThatThrownBy(() -> new Landmarks(new float[8]))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("expected 5 points");
		}

		@Test
		@DisplayName("the geometry check catches swapped eyes and upside-down faces")
		void geometryCheck() {
			assertThat(new Landmarks(SANE).geometryLooksSane()).isTrue();

			float[] eyesSwapped = { 70, 50, 30, 50, 50, 65, 35, 85, 65, 85 };
			assertThat(new Landmarks(eyesSwapped).geometryLooksSane()).isFalse();

			float[] mouthAboveEyes = { 30, 85, 70, 85, 50, 65, 35, 50, 65, 50 };
			assertThat(new Landmarks(mouthAboveEyes).geometryLooksSane()).isFalse();
		}

		@Test
		@DisplayName("scaling all ten coordinates preserves the ordering")
		void scaled() {
			Landmarks s = new Landmarks(SANE).scaled(0.5f);
			assertThat(s.x(0)).isEqualTo(15f);
			assertThat(s.y(4)).isEqualTo(42.5f);
			assertThat(s.geometryLooksSane()).isTrue();
		}
	}

	@Nested
	class Faces {

		@Test
		@DisplayName("similarity is the dot product of two unit vectors")
		void similarity() {
			Face a = face(new float[] { 1, 0, 0, 0 });
			Face b = face(new float[] { 1, 0, 0, 0 });
			Face c = face(new float[] { 0, 1, 0, 0 });
			Face d = face(new float[] { -1, 0, 0, 0 });

			assertThat(a.similarity(b)).isCloseTo(1.0, within(1e-6));
			assertThat(a.similarity(c)).isCloseTo(0.0, within(1e-6));
			assertThat(a.similarity(d)).isCloseTo(-1.0, within(1e-6));
		}

		@Test
		@DisplayName("comparing against a missing or differently sized embedding yields 0, not a crash")
		void similarityIsDefensive() {
			// 128-d SFace and 512-d ArcFace vectors do end up in the same collection. Throwing
			// here would take down a batch job; returning a similarity is worse still.
			Face sface = face(new float[128]);
			Face arcface = face(new float[512]);
			assertThat(sface.similarity(arcface)).isEqualTo(0.0);

			Face noEmbedding = Face.of(new BoundingBox(0, 0, 1, 1), 1f, null);
			assertThat(noEmbedding.similarity(sface)).isEqualTo(0.0);
			assertThat(sface.similarity(noEmbedding)).isEqualTo(0.0);
		}

		@Test
		@DisplayName("hasEmbedding is false for null and for empty")
		void hasEmbedding() {
			assertThat(Face.of(new BoundingBox(0, 0, 1, 1), 1f, null).hasEmbedding()).isFalse();
			assertThat(face(new float[0]).hasEmbedding()).isFalse();
			assertThat(face(new float[] { 1 }).hasEmbedding()).isTrue();
		}

		@Test
		@DisplayName("withEmbedding leaves the original untouched")
		void withEmbeddingIsNonMutating() {
			Face original = Face.of(new BoundingBox(1, 2, 3, 4), 0.5f, null);
			Face embedded = original.withEmbedding(new float[] { 1, 0 });

			assertThat(original.hasEmbedding()).isFalse();
			assertThat(embedded.hasEmbedding()).isTrue();
			assertThat(embedded.box()).isEqualTo(original.box());
			assertThat(embedded.score()).isEqualTo(original.score());
		}

		@Test
		@DisplayName("optionalLandmarks wraps the nullable field")
		void optionalLandmarks() {
			assertThat(Face.of(new BoundingBox(0, 0, 1, 1), 1f, null).optionalLandmarks())
				.isEmpty();
			assertThat(Face.of(new BoundingBox(0, 0, 1, 1), 1f,
				new Landmarks(new float[10])).optionalLandmarks()).isPresent();
		}

		private static Face face(float[] emb) {
			return new Face(new BoundingBox(0, 0, 1, 1), 1f, null, emb);
		}
	}

	@Nested
	class Crops {

		@Test
		@DisplayName("a crop of the wrong size is rejected at construction")
		void sizeIsEnforced() {
			assertThatThrownBy(() -> new AlignedFace(new byte[100], new float[6], null))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("112x112x3");
			assertThatThrownBy(() -> new AlignedFace(bytes(0), new float[4], null))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("6 floats");
		}

		@Test
		@DisplayName("stddev is zero for a flat crop and positive for a varied one")
		void stddev() {
			// The cheapest detector of a warp that sampled entirely outside the source image.
			assertThat(new AlignedFace(bytes(7), new float[6], null).stddev()).isEqualTo(0.0);

			byte[] mixed = bytes(0);
			for (int i = 0; i < mixed.length; i += 2) {
				mixed[i] = (byte) 255;
			}
			assertThat(new AlignedFace(mixed, new float[6], null).stddev()).isGreaterThan(100.0);
		}

		@Test
		@DisplayName("channel access is unsigned")
		void unsignedAccess() {
			byte[] px = bytes(0);
			px[0] = (byte) 200;
			// A signed read here gives -56 and quietly poisons every downstream mean.
			assertThat(new AlignedFace(px, new float[6], null).at(0, 0, 0)).isEqualTo(200);
		}

		private static byte[] bytes(int fill) {
			byte[] b = new byte[AlignedFace.SIZE * AlignedFace.SIZE * 3];
			java.util.Arrays.fill(b, (byte) fill);
			return b;
		}
	}

	@Nested
	class Images {

		@Test
		@DisplayName("a buffer whose length disagrees with the dimensions is rejected")
		void lengthIsEnforced() {
			assertThatThrownBy(() -> FaceImage.ofBgrBytes(4, 4, new byte[10]))
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("width*height*3");
		}

		@Test
		@DisplayName("BGR channel order survives the AWT conversion")
		void bgrOrderFromAwt() {
			// The single most common integration bug in this space: RGB fed to a BGR model still
			// detects faces and still embeds, it just scores worse.
			BufferedImage src = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
			src.setRGB(0, 0, 0x0000FF);   // pure red

			FaceImage img = FaceImage.ofBufferedImage(src);
			assertThat(img.at(0, 0, 0)).as("blue").isEqualTo(0xFF);
			assertThat(img.at(0, 0, 1)).as("green").isEqualTo(0x00);
			assertThat(img.at(0, 0, 2)).as("red").isEqualTo(0x00);
		}

		@Test
		@DisplayName("the TYPE_3BYTE_BGR fast path agrees with the generic path")
		void fastPathAgreesWithSlowPath() {
			// The fast path skips the per-pixel conversion entirely, so the two could disagree
			// without anything failing.
			BufferedImage rgb = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB);
			for (int y = 0; y < 2; y++) {
				for (int x = 0; x < 3; x++) {
					rgb.setRGB(x, y, 0x102030 + x * 7 + y * 11);
				}
			}
			BufferedImage bgr = new BufferedImage(3, 2, BufferedImage.TYPE_3BYTE_BGR);
			bgr.getGraphics().drawImage(rgb, 0, 0, null);

			assertThat(FaceImage.ofBufferedImage(bgr).bgr())
				.isEqualTo(FaceImage.ofBufferedImage(rgb).bgr());
		}

		@Test
		@DisplayName("the fast path copies rather than aliasing the source raster")
		void fastPathCopies() {
			// Without the clone, mutating the BufferedImage afterwards would silently change
			// pixels an inference call is already reading.
			BufferedImage src = new BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR);
			FaceImage img = FaceImage.ofBufferedImage(src);
			src.setRGB(0, 0, 0xFFFFFF);
			assertThat(img.at(0, 0, 0)).isEqualTo(0);
		}

		@Test
		@DisplayName("pixel access is row-major with stride width*3")
		void layout() {
			byte[] buf = new byte[2 * 2 * 3];
			buf[(1 * 2 + 1) * 3 + 2] = (byte) 0xAB;   // x=1, y=1, red
			FaceImage img = FaceImage.ofBgrBytes(2, 2, buf);
			assertThat(img.at(1, 1, 2)).isEqualTo(0xAB);
			assertThat(img.at(0, 0, 2)).isEqualTo(0);
		}
	}

	@Nested
	class Devices {

		@Test
		@DisplayName("cuda is the default and cpu must be asked for")
		void factories() {
			assertThat(Device.cuda().isCuda()).isTrue();
			assertThat(Device.cuda().ordinal()).isEqualTo(0);
			assertThat(Device.cuda(3).ordinal()).isEqualTo(3);
			assertThat(Device.cpu().isCuda()).isFalse();
		}

		@Test
		@DisplayName("toString is the form that goes into logs and metrics")
		void rendering() {
			assertThat(Device.cuda().toString()).isEqualTo("cuda:0");
			assertThat(Device.cuda(1).toString()).isEqualTo("cuda:1");
			assertThat(Device.cpu().toString()).isEqualTo("cpu");
		}
	}
}
