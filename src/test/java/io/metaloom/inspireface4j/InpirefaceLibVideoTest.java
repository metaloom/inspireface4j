package io.metaloom.inspireface4j;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class InpirefaceLibVideoTest extends AbstractInspireFaceLibTest {

	@Test
	public void testVideo() throws Throwable {
		SimpleImageViewer viewer = new SimpleImageViewer();

		// try (VideoFile video = VideoFile.open("src/test/resources/8090198-hd_1366_720_25fps.mp4")) {
		try (VideoFile video = VideoFile.open("/extra/vid/4.mkv")) {
			video.seekToFrameRatio(0.3);
			long start = System.currentTimeMillis();

			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				Mat imageMat = frame.mat();
				FaceDetections detections = InspirefaceLib.detect(imageMat, false);
				if (!detections.isEmpty()) {
					//InspirefaceLib.embedding(imageMat, detections, 0);
					List<FaceAttributes> attrs = InspirefaceLib.attributes(imageMat, detections, true);
				}
				// if (attrs.size() >= 1) {
				// System.out.println(attrs.getFirst());
				// }
				drawDetections(detections, imageMat);

				viewer.show(imageMat);
			}
			long dur = System.currentTimeMillis() - start;
			System.out.println("Took " + dur);
			Thread.sleep(50);
		}

	}

	private void drawDetections(List<Detection> detections, Mat imageMat) {
		for (Detection det : detections) {
			BoundingBox box = det.box();
			CVUtils.drawRect(imageMat, box.x, box.y, box.width, box.height);
		}

	}
}
