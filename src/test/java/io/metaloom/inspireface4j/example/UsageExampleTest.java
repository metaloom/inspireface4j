package io.metaloom.inspireface4j.example;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import static io.metaloom.inspireface4j.SessionFeature.*;
import io.metaloom.inspireface4j.BoundingBox;
import io.metaloom.inspireface4j.Detection;
import io.metaloom.inspireface4j.FaceAttributes;
import io.metaloom.inspireface4j.InspirefaceLib;
import io.metaloom.inspireface4j.InspirefaceSession;
import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.video4j.Video4j;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.ImageUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class UsageExampleTest {

	private static final String DEFAULT_PACK = "packs/Pikachu";

	@Test
	public void testImageUsageExample() throws IOException, InterruptedException {
		// SNIPPET START image-usage.example
		String imagePath = "src/test/resources/pexels-olly-3812743_1k_lower.jpg";

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();

		try (InspirefaceSession session = InspirefaceLib.session(DEFAULT_PACK, 640,  ENABLE_FACE_RECOGNITION, ENABLE_FACE_ATTRIBUTE)) {

			// Load the image and invoke the detection
			BufferedImage img = ImageUtils.load(new File(imagePath));
			Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
			CVUtils.bufferedImageToMat(img, imageMat);

			// Invoke the detection
			FaceDetections detections = session.detect(imageMat, true);

			// Print the detections
			for (Detection detection : detections) {
				System.out.println(detection.box() + " @ " + String.format("%.4f", detection.conf()));
			}
			// Show the result and release the mat and the detection data
			ImageUtils.show(imageMat);
			MatProvider.released(imageMat);
		}
		// SNIPPET END image-usage.example
		Thread.sleep(2000);
	}

	@Test
	public void testVideoUsageExample() throws IOException {
		// SNIPPET START video-usage.example

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();
		SimpleImageViewer viewer = new SimpleImageViewer();

		
		try (InspirefaceSession session = InspirefaceLib.session("packs/Pikachu", 640, ENABLE_FACE_RECOGNITION, ENABLE_FACE_ATTRIBUTE)) {

			// Open the video using Video4j
			try (VideoFile video = VideoFile.open("src/test/resources/8090198-hd_1366_720_25fps.mp4")) {

				// Process each frame
				VideoFrame frame;
				while ((frame = video.frame()) != null) {
					// System.out.println(frame);

					// Optionally downscale the frame
					CVUtils.resize(frame, 512);

					// Run the detection on the mat reference
					FaceDetections detections = session.detect(frame.mat(), true);

					if (!detections.isEmpty()) {
						// Extract the face embedding from the first face
						float[] embedding = session.embedding(frame.mat(), detections, 0);
						// Extract the face attributes
						List<FaceAttributes> attrs = session.attributes(frame.mat(), detections, true);
					}

					// Print the detections
					for (Detection detection : detections) {
						double confidence = detection.conf();
						BoundingBox box = detection.box();
						System.out.println("Frame[" + video.currentFrame() + "] = " + confidence + " @ " + box);
					}

					viewer.show(frame.mat());
				}
			}
		}

		// SNIPPET END video-usage.example
	}

	@Test
	public void testVideoPerformance() throws IOException {
		long start = System.currentTimeMillis();
		long nFrames = 0;

		// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
		Video4j.init();
		InspirefaceLib.init("packs/Pikachu", 640);

		// Open the video using Video4j
		try (VideoFile video = VideoFile.open("src/test/resources/8090198-hd_1366_720_25fps.mp4")) {

			// Process each frame
			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				// Run the detection on the mat reference
				InspirefaceLib.detect(null, frame.mat(), false);
				nFrames++;

			}
		}
		long dur = System.currentTimeMillis() - start;
		double avg = (double) dur / (double) nFrames;
		System.out.println("Took: " + dur + " ms for " + nFrames + " frames. Avg: " + String.format("%.2f", avg) + "ms per frame");
	}

}
