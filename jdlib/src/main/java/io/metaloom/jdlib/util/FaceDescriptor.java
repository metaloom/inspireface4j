package io.metaloom.jdlib.util;

import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;

public class FaceDescriptor {

	private final Rectangle faceBox;
	private final List<Point> facialLandmarks;
	private final float[] faceEmbedding;
	private String label;

	public FaceDescriptor(Rectangle faceBoxes, List<Point> facialLandmarks) {
		this.faceBox = faceBoxes;
		this.facialLandmarks = facialLandmarks;
		this.faceEmbedding = null;
	}

	public FaceDescriptor(Rectangle faceBoxes, List<Point> facialLandmarks, float[] faceEmbedding) {
		this.faceBox = faceBoxes;
		this.facialLandmarks = facialLandmarks;
		this.faceEmbedding = faceEmbedding;
		this.label = "None";
	}

	public Rectangle getFaceBox() {
		return faceBox;
	}

	public List<Point> getFacialLandmarks() {
		return facialLandmarks;
	}

	public float[] getFaceEmbedding() {
		return faceEmbedding;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

}
