package io.metaloom.inspireface4j;

import java.awt.image.BufferedImage;
import java.lang.foreign.MemorySegment;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;

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

	/**
	 * Extract the face embedding from the provided image. Note that a more efficient method exists that directly uses {@link Mat} data as image source.
	 * 
	 * @param mat
	 * @param detections
	 * @param faceNr
	 * @return
	 */
	public float[] embedding(BufferedImage image, FaceDetections detections, int faceNr) {
		Mat imageMat = MatProvider.mat(image, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(image, imageMat);
		float[] embedding = InspirefaceLib.embedding(this, imageMat, detections, faceNr);
		MatProvider.released(imageMat);
		return embedding;
	}

	/**
	 * Extract the face embedding from the provided image.
	 * 
	 * @param mat
	 * @param detections
	 * @param faceNr
	 * @return
	 */
	public float[] embedding(Mat mat, FaceDetections detections, int faceNr) {
		return InspirefaceLib.embedding(this, mat, detections, faceNr);
	}

	public List<FaceAttributes> attributes(Mat mat, FaceDetections detections, boolean drawAttributes) {
		return InspirefaceLib.attributes(this, mat, detections, drawAttributes);
	}

	protected MemorySegment data() {
		return sessionPtr;
	}

	public  List<FaceLandmark> landmarks(Mat mat, FaceDetections detections, int faceNr, boolean drawLandmarks) {
		return InspirefaceLib.landmarks(this, mat, detections, faceNr, drawLandmarks);
	}

}
