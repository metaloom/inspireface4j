package io.metaloom.facedetect4j.yunet.example;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.GraphicsEnvironment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.BoundingBox;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.facedetect4j.api.TestData;
import io.metaloom.facedetect4j.yunet.GpuProbe;
import io.metaloom.facedetect4j.yunet.Yunet4j;
import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.core.Scalar;
import io.metaloom.video4j.Video4j;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

/**
 * The examples that the README is generated from. They are tests rather than a doc comment so that
 * a signature change breaks the build instead of quietly rotting the documentation.
 *
 * <p>
 * Everything the reader does not need to see - locating the models and the test photos - is kept
 * outside the SNIPPET markers, so the README shows the API and nothing else.
 */
public class UsageExampleTest {

	private static final Path MODELS = Path.of("models");

	private static final Path photo = TestData.image(TestData.IMG_FACE_NEUTRAL);
	private static final Path otherPhoto = TestData.image(TestData.IMG_FACE_HAPPY);
	private static final Path clip = TestData.video(TestData.VID_FACE_ROTATE_1);

	private static void requireModelsAndGpu() {
		// The examples construct a GPU pipeline, which throws by design when CUDA will not work.
		assumeTrue(GpuProbe.usable(MODELS), GpuProbe.skipReason());
	}

	@Test
	public void testDetectionUsageExample() throws Exception {
		requireModelsAndGpu();

		// SNIPPET START detect-usage.example
		// GPU 0. Throws if the CUDA provider will not attach - it never falls back silently.
		try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"))) {

			FaceImage img = FaceImage.read(photo);

			// Every face above threshold, unsorted and uncapped
			for (Face face : faces.detect(img)) {
				System.out.println(face.box() + " @ " + String.format("%.4f", face.score()));
			}
		}
		// SNIPPET END detect-usage.example
	}

	@Test
	public void testEmbeddingUsageExample() throws Exception {
		requireModelsAndGpu();

		// SNIPPET START embed-usage.example
		try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"), Device.cuda())) {

			FaceImage img = FaceImage.read(photo);

			// Detection and embedding in one pass
			for (Face face : faces.detectAndEmbed(img)) {
				System.out.println(face.box() + " -> " + face.embedding().length + "d");
			}

			// Or pick the single face a portrait pipeline wants
			Face a = faces.primaryFace(img, faces.detect(img)).orElseThrow();
			a = a.withEmbedding(faces.embed(img, a));

			FaceImage other = FaceImage.read(otherPhoto);
			Face b = faces.primaryFace(other, faces.detect(other)).orElseThrow();
			b = b.withEmbedding(faces.embed(other, b));

			// Embeddings are L2 normalised, so comparing is a plain dot product:
			// ~0.65 for the same person, ~0.16 for different people
			System.out.println("similarity: " + a.similarity(b));

			// Align once, embed from the crop - the entry point that lets two embedders be
			// compared on identical pixels
			AlignedFace crop = faces.align(img, a);
			float[] embedding = faces.embed(crop);
			System.out.println("dimensions: " + embedding.length);
		}
		// SNIPPET END embed-usage.example
	}

	/**
	 * The same pipeline over a video, one frame at a time.
	 *
	 * <p>
	 * video4j owns the decoding and hands out an OpenCV {@code Mat}, and this module still has no
	 * OpenCV on its compile path — the dependency is test-scoped. The whole bridge is
	 * {@link #toFaceImage}: {@code CV_8UC3} is already BGR, row-major, stride {@code width * 3},
	 * which is exactly what {@link FaceImage} documents, so it is a copy rather than a conversion.
	 */
	@Test
	public void testVideoUsageExample() throws Exception {
		requireModelsAndGpu();
		assumeTrue(Files.isRegularFile(clip), "test clip missing: " + clip);
		// The example opens a window, which is the point of it. Skipped rather than failed on a
		// headless machine, where SimpleImageViewer's JFrame throws before the first frame.
		assumeFalse(GraphicsEnvironment.isHeadless(), "no display -- this example shows a window");

		// SNIPPET START video-usage.example
		Video4j.init();
		SimpleImageViewer viewer = new SimpleImageViewer();

		Face reference = null;
		List<Double> identity = new ArrayList<>();

		try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"));
			VideoFile video = VideoFile.open(clip)) {

			VideoFrame decoded;
			while ((decoded = video.frame()) != null) {

				// try-with-resources releases the frame's Mat. video4j allocates a fresh one per
				// frame and native memory is invisible to the garbage collector, so a long clip
				// processed without this grows until the OS intervenes rather than the JVM.
				try (VideoFrame frame = decoded) {

					// No resize step: the detector caps the long edge itself
					// (YuNetDetector.setMaxInputEdge), so one here would only add a second
					// interpolation and move the boxes.
					FaceImage img = toFaceImage(frame.mat());

					List<Face> detections = faces.detect(img);

					// Print the detections, and draw them onto the frame the viewer shows.
					// Drawing after the conversion above, so the boxes never reach the model.
					for (Face detection : detections) {
						BoundingBox box = detection.box();
						System.out.println("Frame[" + video.currentFrame() + "] = "
							+ detection.score() + " @ " + box);
						CVUtils.drawRect(frame.mat(), (int) box.x1(), (int) box.y1(),
							(int) box.width(), (int) box.height(), new Scalar(0, 255, 0));
					}

					// The single face a portrait pipeline wants, embedded and compared against
					// the first frame's. Embeddings from separate calls are directly comparable,
					// so "is this still the same person" needs no state beyond one reference
					// vector — no tracker, no frame-to-frame association.
					Optional<Face> found = faces.primaryFace(img, detections);
					if (found.isPresent()) {
						Face face = found.get().withEmbedding(faces.embed(img, found.get()));
						if (reference == null) {
							reference = face;
						}
						identity.add(reference.similarity(face));
					}

					viewer.show(frame.mat());
				}
			}
		}

		// Judge a clip on the distribution, not on the worst frame. This one is a head turning
		// through full profile, and at profile SFace sees one eye and no mouth corners: the
		// cosine against a frontal reference falls to roughly zero on ~10% of frames while the
		// detector is still reporting 0.8. Per-frame thresholding would call that a stranger.
		Collections.sort(identity);
		System.out.printf("a face in %d frames -- identity median %.3f, p10 %.3f, worst %.3f%n",
			identity.size(), identity.get(identity.size() / 2),
			identity.get(identity.size() / 10), identity.get(0));
		// SNIPPET END video-usage.example

		assertThat(identity).as("frames with a detected face").hasSizeGreaterThan(50);
		assertThat(identity.get(identity.size() / 2)).as("median same-identity score")
			.isGreaterThan(0.6);
	}

	// SNIPPET START video-usage.bridge
	/** OpenCV {@code CV_8UC3} is BGR, row-major, stride {@code width * 3} — FaceImage's layout. */
	private static FaceImage toFaceImage(Mat mat) {
		byte[] bgr = new byte[mat.rows() * mat.cols() * 3];
		int copied = mat.get(0, 0, bgr);
		// A non-continuous or non-8UC3 Mat copies short and leaves the tail zeroed, which detects
		// as a plausible-looking nothing rather than as an error. Cheap to rule out here.
		if (copied != bgr.length) {
			throw new IllegalArgumentException("copied " + copied + " of " + bgr.length
				+ " bytes -- Mat is not continuous CV_8UC3");
		}
		return FaceImage.ofBgrBytes(mat.cols(), mat.rows(), bgr);
	}
	// SNIPPET END video-usage.bridge
}
