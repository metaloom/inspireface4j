package io.metaloom.inspireface4j;

public class Detection {

	private BoundingBox box;
	private float conf;
	private FaceAttributes attributes;
	private FaceAngles angles;

	public Detection(BoundingBox box, float conf, FaceAngles angles) {
		this.box = box;
		this.conf = conf;
		this.angles = angles;
	}

	public BoundingBox box() {
		return box;
	}

	public float conf() {
		return conf;
	}

	public FaceAttributes attributes() {
		return attributes;
	}

	public FaceAngles angles() {
		return angles;
	}

	public void setAttributes(FaceAttributes attributes) {
		this.attributes = attributes;
	}

}
