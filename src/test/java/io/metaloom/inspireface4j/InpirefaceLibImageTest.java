package io.metaloom.inspireface4j;

import static io.metaloom.inspireface4j.SessionFeature.ENABLE_FACE_ATTRIBUTE;
import static io.metaloom.inspireface4j.SessionFeature.ENABLE_FACE_RECOGNITION;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.inspireface4j.data.internal.HFLogLevel;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.ImageUtils;

public class InpirefaceLibImageTest extends AbstractInspireFaceLibTest {

	private static String imagePath = TestMedia.IMG_FACE_RASTER_1K_UPPER;

	@Test
	public void testImage() throws Throwable {
		BufferedImage img = ImageUtils.load(new File(imagePath));
		assertNotNull(img);
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);

		System.out.println("Detect");
		try (InspirefaceSession session = InspirefaceLib.session("packs/Pikachu", 640, ENABLE_FACE_ATTRIBUTE, ENABLE_FACE_RECOGNITION)) {
			detect(session, imageMat);
		}
		Thread.sleep(9000);
		// System.in.read();
	}

	@Test
	public void testMultiSessions() throws InterruptedException, IOException {
		BufferedImage img = ImageUtils.load(new File(imagePath));
		assertNotNull(img);

		System.out.println("Detect");
		try (InspirefaceSession session = InspirefaceLib.session("packs/Pikachu", 640, ENABLE_FACE_ATTRIBUTE, ENABLE_FACE_RECOGNITION)) {
			Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
			CVUtils.bufferedImageToMat(img, imageMat);
			detect(session, imageMat);
		}
		try (InspirefaceSession session = InspirefaceLib.session("packs/Pikachu", 640, ENABLE_FACE_ATTRIBUTE, ENABLE_FACE_RECOGNITION)) {
			Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
			CVUtils.bufferedImageToMat(img, imageMat);
			detect(session, imageMat);
		}
		Thread.sleep(8000);
		// System.in.read();
	}

	private void detect(InspirefaceSession session, Mat imageMat) {
		FaceDetections detections = session.detect(imageMat, true);
		assertNotNull(detections);

		InspirefaceLib.logLevel(HFLogLevel.HF_LOG_DEBUG);
		session.attributes(imageMat, detections, true);
		session.embedding(imageMat, detections, 1);
		for (int i = 0; i < detections.size(); i++) {
			List<FaceLandmark> landmarks = session.landmarks(imageMat, detections, i, true);
			assertFalse(landmarks.isEmpty());
		}
		ImageUtils.show(imageMat);

		for (Detection detection : detections) {
			System.out.println("conf: " + detection.conf());
		}

	}

}
