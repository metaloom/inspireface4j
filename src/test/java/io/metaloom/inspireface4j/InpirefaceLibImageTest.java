package io.metaloom.inspireface4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import io.metaloom.inspireface4j.data.HFLogLevel;
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
		List<Detection> detections = InspirefaceLib.detect(imageMat, true);
		InspirefaceLib.logLevel(HFLogLevel.HF_LOG_DEBUG);
		InspirefaceLib.attributes(imageMat, true);
		InspirefaceLib.embedding(imageMat);
		assertNotNull(detections);
		// assertEquals(3, detections.size());
		ImageUtils.show(imageMat);

		for (Detection detection : detections) {
			System.out.println("conf: " + detection.conf());
		}

		Thread.sleep(1000);
		// System.in.read();
	}

}
