package io.metaloom.facedetect4j.yunet;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.facedetect4j.api.Landmarks;
import io.metaloom.facedetect4j.api.TestData;

/**
 * End-to-end tests against the real models on the real GPU.
 *
 * <p>
 * Golden box values were produced by <b>OpenCV's own</b> {@code FaceDetectorYN} on the same weights
 * at the same resolution — not by a second implementation of the same maths. That is what makes
 * them meaningful: a decoder bug that happens to be self-consistent still fails these.
 *
 * <p>
 * Skipped rather than failed when models or a GPU are absent, so a checkout without the (git-lfs,
 * externally hosted) weights still builds.
 */
class Yunet4jTest {

	private static final Path MODELS = Path.of("models");

	/**
	 * Both the weights and a working CUDA provider. The provider is probed rather than assumed:
	 * ONNX Runtime lists CUDA among its providers on any machine, so a build-level check passes on
	 * a host where the runtime cannot actually be loaded and every test then errors instead of
	 * skipping.
	 */
	@BeforeAll
	static void requireModelsAndGpu() {
		// The weights themselves, not just the directory: they are commonly symlinked in, and a
		// dangling link would otherwise surface as a failure in every test rather than a skip.
		assumeTrue(Files.isReadable(MODELS.resolve(Yunet4j.YUNET_DYNAMIC))
			|| Files.isReadable(MODELS.resolve(Yunet4j.YUNET_FIXED)),
			"no YuNet weights in models/ -- see Yunet4j.downloadHint()");
		assumeTrue(GpuProbe.usable(MODELS), GpuProbe.skipReason());
	}

	private static FacePipeline pipeline() {
		return Yunet4j.pipeline(MODELS);
	}

	private static FaceImage image(String name) throws Exception {
		Path p = TestData.image(name);
		assumeTrue(Files.isRegularFile(p), "test image missing: " + p);
		return FaceImage.read(p);
	}

	@Test
	@DisplayName("runs on the GPU, and says so")
	void runsOnGpu() {
		try (FacePipeline p = pipeline()) {
			assertThat(p.device().isCuda()).isTrue();
			assertThat(p.dimensions()).isEqualTo(128);
		}
	}

	@Test
	@DisplayName("box and score match OpenCV FaceDetectorYN on the same weights")
	void matchesOpenCvReference() throws Exception {
		try (FacePipeline p = pipeline()) {
			// cv2.FaceDetectorYN, face_detection_yunet_2026may.onnx, native resolution.
			assertBox(p, TestData.IMG_FACE_NEUTRAL, 211.5, 147.4, 468.6, 502.7, 0.9491);
			assertBox(p, TestData.IMG_FACE_HAPPY, 194.0, 84.7, 481.1, 482.3, 0.9475);
		}
	}

	private static void assertBox(FacePipeline p, String img,
		double x1, double y1, double x2, double y2, double score) throws Exception {
		FaceImage image = image(img);
		Face f = p.primaryFace(image, p.detect(image)).orElseThrow();
		// 1 px: yunet4j pads the input up to a 32 multiple (mandatory -- the model cannot run
		// otherwise under ORT) while OpenCV runs the ragged size, so exact equality is not
		// available. Drift beyond a pixel would mean a decode error, not padding.
		assertThat((double) f.box().x1()).as("%s x1", img).isCloseTo(x1, within(1.0));
		assertThat((double) f.box().y1()).as("%s y1", img).isCloseTo(y1, within(1.0));
		assertThat((double) f.box().x2()).as("%s x2", img).isCloseTo(x2, within(1.0));
		assertThat((double) f.box().y2()).as("%s y2", img).isCloseTo(y2, within(1.0));
		assertThat((double) f.score()).as("%s score", img).isCloseTo(score, within(0.01));
	}

	private static org.assertj.core.data.Offset<Double> within(double d) {
		return org.assertj.core.data.Offset.offset(d);
	}

	@Test
	@DisplayName("keypoints come back in ArcFace order")
	void keypointOrder() throws Exception {
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			Face f = p.primaryFace(img, p.detect(img)).orElseThrow();
			Landmarks lm = f.landmarks();
			assertThat(lm).isNotNull();
			// A permuted order still produces a face-shaped crop that embeds without error and
			// merely scores worse, so this is checked explicitly rather than assumed.
			assertThat(lm.geometryLooksSane()).as("left eye left of right eye, eyes above mouth")
				.isTrue();
		}
	}

	@Test
	@DisplayName("separates identities: same person scores far above different people")
	void identitySeparation() throws Exception {
		try (FacePipeline p = pipeline()) {
			Face a = embedded(p, TestData.IMG_FACE_NEUTRAL);
			Face b = embedded(p, TestData.IMG_FACE_HAPPY);
			Face c = embedded(p, TestData.IMG_FACE_OTHER_IDENTITY);

			double same = a.similarity(b);
			double diff = a.similarity(c);
			assertThat(same).as("same identity, different expression").isGreaterThan(0.45);
			assertThat(diff).as("different identity").isLessThan(0.30);
			assertThat(same - diff).as("margin").isGreaterThan(0.25);
		}
	}

	private static Face embedded(FacePipeline p, String name) throws Exception {
		FaceImage img = image(name);
		Face f = p.primaryFace(img, p.detect(img)).orElseThrow();
		return f.withEmbedding(p.embed(img, f));
	}

	@Test
	@DisplayName("embeddings are unit length, so similarity is a plain dot product")
	void embeddingsAreNormalised() throws Exception {
		try (FacePipeline p = pipeline()) {
			Face f = embedded(p, TestData.IMG_FACE_NEUTRAL);
			double n = 0;
			for (float v : f.embedding()) {
				n += (double) v * v;
			}
			assertThat(Math.sqrt(n)).isCloseTo(1.0, within(1e-4));
		}
	}

	@Test
	@DisplayName("alignment produces a usable 112x112 crop")
	void alignmentProducesCrop() throws Exception {
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_FACE_NEUTRAL);
			Face f = p.primaryFace(img, p.detect(img)).orElseThrow();
			AlignedFace crop = p.align(img, f);
			assertThat(crop.bgr()).hasSize(112 * 112 * 3);
			// Near-zero variance means the warp sampled entirely outside the image.
			assertThat(crop.stddev()).isGreaterThan(5.0);
			// Embedding from the pre-aligned crop must equal the convenience path.
			assertThat(p.embed(crop)).isEqualTo(p.embed(img, f));
		}
	}

	@Test
	@DisplayName("finds every face in a crowded image")
	void multiFace() throws Exception {
		try (FacePipeline p = pipeline()) {
			FaceImage img = image(TestData.IMG_COLLAGE_24_FACES);
			List<Face> faces = p.detect(img);
			// Ground truth is 24; detectors legitimately differ by a couple on the boundary cases.
			assertThat(faces).hasSizeBetween(20, 30);
			assertThat(faces).allSatisfy(f -> assertThat(f.score()).isGreaterThan(0.5f));
		}
	}

	@Test
	@DisplayName("a missing model names the file and how to get it")
	void missingModelIsActionable() {
		assertThatThrownBy(() -> Yunet4j.pipeline(Path.of("does-not-exist")))
			.isInstanceOf(FaceException.class)
			.hasMessageContaining("media.githubusercontent.com");
	}
}
