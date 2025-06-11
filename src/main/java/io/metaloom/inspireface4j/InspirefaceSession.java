package io.metaloom.inspireface4j;

import java.lang.foreign.MemorySegment;
import java.util.List;

import org.opencv.core.Mat;

import io.metaloom.inspireface4j.data.FaceDetections;

public class InspirefaceSession implements AutoCloseable {

	private MemorySegment sessionPtr;

	public InspirefaceSession(MemorySegment sessionPtr) {
		this.sessionPtr = sessionPtr;
	}

	@Override
	public void close() throws Inspireface4jException {
		InspirefaceLib.releaseSession(sessionPtr);
	}

	public FaceDetections detect(Mat mat, boolean drawDetections) {
		return InspirefaceLib.detect(this, mat, drawDetections);
	}

	public float[] embedding(Mat mat, FaceDetections detections, int faceNr) {
		return InspirefaceLib.embedding(this, mat, detections, faceNr);
	}

	public List<FaceAttributes> attributes(Mat mat, FaceDetections detections, boolean drawAttributes) {
		return InspirefaceLib.attributes(this, mat, detections, drawAttributes);
	}

	protected MemorySegment data() {
		return sessionPtr;
	}

}
