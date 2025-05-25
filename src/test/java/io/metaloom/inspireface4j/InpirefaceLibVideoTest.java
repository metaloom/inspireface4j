package io.metaloom.inspireface4j;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class InpirefaceLibVideoTest extends AbstractInspireFaceLibTest {

	@Test
	public void testVideo() throws Throwable {
		SimpleImageViewer viewer = new SimpleImageViewer();

		// try (VideoFile video = VideoFile.open("src/test/resources/8090198-hd_1366_720_25fps.mp4")) {
		try (VideoFile video = VideoFile.open("/extra/vid/1.avi")) {
			// video.seekToFrameRatio(0.1);
			long start = System.currentTimeMillis();

			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				Mat imageMat = frame.mat();
				List<Detection> detections = InspirefaceLib.detect(imageMat, false);
				drawDetections(detections, imageMat);
				viewer.show(imageMat);
			}
			long dur = System.currentTimeMillis() - start;
			System.out.println("Took " + dur);
			Thread.sleep(100);
		}

	}

	private void drawDetections(List<Detection> detections, Mat imageMat) {
		for (Detection det : detections) {
			BoundingBox box = det.box();
			CVUtils.drawRect(imageMat, box.x, box.y, box.width, box.height);
		}

	}
}
