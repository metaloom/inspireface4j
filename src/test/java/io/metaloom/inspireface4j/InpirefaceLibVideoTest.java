package io.metaloom.inspireface4j;

import static io.metaloom.inspireface4j.SessionFeature.ENABLE_FACE_ATTRIBUTE;
import static io.metaloom.inspireface4j.SessionFeature.ENABLE_FACE_POSE;
import static io.metaloom.inspireface4j.SessionFeature.ENABLE_FACE_RECOGNITION;

import java.util.List;

import org.junit.jupiter.api.Test;
import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.core.Point;
import io.metaloom.opencv.core.Scalar;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class InpirefaceLibVideoTest extends AbstractInspireFaceLibTest {

	@Test
	public void testVideo() throws Throwable {
		SimpleImageViewer viewer = new SimpleImageViewer();

		try (InspirefaceSession session = InspirefaceLib.session("packs/Pikachu", 640, ENABLE_FACE_ATTRIBUTE, ENABLE_FACE_RECOGNITION,
			ENABLE_FACE_POSE)) {

			try (VideoFile video = VideoFile.open(TestMedia.VID_FACE_ROTATE_1)) {
				video.seekToFrameRatio(0.1);
				long start = System.currentTimeMillis();

				VideoFrame frame;
				while ((frame = video.frame()) != null) {
					Mat imageMat = frame.mat();
					FaceDetections detections = session.detect(imageMat, false);
					if (!detections.isEmpty()) {
						// InspirefaceLib.embedding(imageMat, detections, 0);
						session.attributes(imageMat, detections, true);
						for (int i = 0; i < detections.size(); i++) {
							session.landmarks(imageMat, detections, 0, true);
						}
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

	}

	private void drawDetections(List<Detection> detections, Mat imageMat) {
		for (Detection det : detections) {
			BoundingBox box = det.box();
			CVUtils.drawRect(imageMat, box.x, box.y, box.width, box.height);

			Point p = new Point(box.x, box.y + 15);
			Scalar color = new Scalar(255f, 255f, 0f);
			float fontWeight = 1.0f;
			int thickness = 2;
			CVUtils.drawText(imageMat, String.format("A %.2f,%.2f,%.2f", det.angles().roll(), det.angles().yaw(), det.angles().pitch()), p,
				fontWeight, color, thickness);
		}

	}
}
