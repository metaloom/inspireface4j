package io.metaloom.inspireface4j.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.jupiter.api.Test;

import io.metaloom.inspireface4j.BoundingBox;
import io.metaloom.inspireface4j.Detection;
import io.metaloom.inspireface4j.InspirefaceLib;
import io.metaloom.video4j.Video4j;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.ImageUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class UsageExampleTest {

	@Test
	public void testImageUsageExample() throws IOException {
		// SNIPPET START image-usage.example
		String imagePath = "insightface/cpp-package/inspireface/test_res/data/crop/crop.png";
		boolean useGPU = true;

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();
		InspirefaceLib.init("insightface/cpp-package/inspireface/test_res/pack/Megatron", useGPU);
		
		// Load the image and invoke the detection
		BufferedImage img = ImageUtils.load(new File(imagePath));
		List<Detection> detections = InspirefaceLib.detect(img, false);

		// Print the detections
		for (Detection detection : detections) {
			System.out.println(detection.label() + " = " + detection.conf() + " @ " + detection.box());
		}
		// SNIPPET END image-usage.example
	}

	@Test
	public void testVideoUsageExample() throws IOException {
		// SNIPPET START video-usage.example
		boolean useGPU = false;

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();
		InspirefaceLib.init("insightface/cpp-package/inspireface/test_res/pack/Megatron", useGPU);
		SimpleImageViewer viewer = new SimpleImageViewer();

		// Open the video using Video4j
		try (VideoFile video = VideoFile.open("src/test/resources/3769953-hd_1920_1080_25fps.mp4")) {

			// Process each frame
			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				System.out.println(frame);
				CVUtils.resize(frame, 1024);
				// Run the detection on the mat reference
				List<Detection> detections = InspirefaceLib.detect(frame.mat(), true);

				// Print the detections
				for (Detection detection : detections) {
					String label = detection.label();
					double confidence = detection.conf();
					BoundingBox box = detection.box();
					System.out.println("Frame[" + video.currentFrame() + "] " + label + " = " + confidence + " @ " + box);
				}

				viewer.show(frame.mat());
			}
		}
		// SNIPPET END video-usage.example
	}

	@Test
	public void testVideoPerformance() throws IOException {
		long start = System.currentTimeMillis();
		long nFrames = 0;
		// SNIPPET START video-usage.example
		boolean useGPU = false;

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();
		InspirefaceLib.init("insightface/cpp-package/inspireface/test_res/pack/Megatron", useGPU);

		// Open the video using Video4j
		try (VideoFile video = VideoFile.open("src/test/resources/3769953-hd_1920_1080_25fps.mp4")) {

			// Process each frame
			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				// Run the detection on the mat reference
				InspirefaceLib.detect(frame.mat(), false);
				nFrames++;

			}
		}
		// SNIPPET END video-usage.example
		long dur = System.currentTimeMillis() - start;
		double avg = (double) dur / (double) nFrames;
		System.out.println("Took: " + dur + " ms for " + nFrames + " frames. Avg: " + String.format("%.2f", avg));
	}

}
