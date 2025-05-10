package io.metaloom.inspireface4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import io.metaloom.video4j.Video4j;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.ImageUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class InpirefaceLibTest {

	private static String imagePath = "insightface/cpp-package/inspireface/test_res/data/crop/crop.png";

	private static String modelPath = "insightface/cpp-package/inspireface/test_res/pack/Megatron";
	
	static {
		Video4j.init();
		InspirefaceLib.init(modelPath, false);
	}

	@Test
	public void testImage() throws Throwable {
		// System.setProperty("java.library.path", onnxLibPath);
		BufferedImage img = ImageUtils.load(new File(imagePath));
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		List<Detection> detections = InspirefaceLib.detect(imageMat, true);
		assertNotNull(detections);
		assertEquals(3, detections.size());
		ImageUtils.show(imageMat);

		for (Detection detection : detections) {
			System.out.println(detection.label() + " conf: " + detection.conf());
		}

		System.in.read();
	}

	@Test
	public void testVideo() throws Throwable {
		SimpleImageViewer viewer = new SimpleImageViewer();

		try (VideoFile video = VideoFile.open("src/test/resources/3769953-hd_1920_1080_25fps.mp4")) {
			video.seekToFrameRatio(0.5);
			long start = System.currentTimeMillis();

			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				Mat imageMat = frame.mat();
				InspirefaceLib.detect(imageMat, true);
				viewer.show(imageMat);
			}
			long dur = System.currentTimeMillis() - start;
			System.out.println("Took " + dur);
			Thread.sleep(100);
		}

	}
}
