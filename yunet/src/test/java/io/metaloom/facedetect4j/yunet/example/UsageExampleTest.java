package io.metaloom.facedetect4j.yunet.example;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.facedetect4j.api.TestData;
import io.metaloom.facedetect4j.yunet.Yunet4j;

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

	private static void requireModels() {
		assumeTrue(Files.isReadable(MODELS.resolve(Yunet4j.SFACE)),
			"models/ not populated -- see Yunet4j.downloadHint()");
	}

	@Test
	public void testDetectionUsageExample() throws Exception {
		requireModels();

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
		requireModels();

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
}
