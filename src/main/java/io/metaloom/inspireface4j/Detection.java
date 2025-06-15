package io.metaloom.inspireface4j;

public class Detection {

	private BoundingBox box;
	private float conf;
	private FaceAttributes attributes;

	public Detection(BoundingBox box, float conf) {
		this.box = box;
		this.conf = conf;
	}

	public BoundingBox box() {
		return box;
	}

	public float conf() {
		return conf;
	}

	public FaceAttributes getAttributes() {
		return attributes;
	}

	public void setAttributes(FaceAttributes attributes) {
		this.attributes = attributes;
	}

}
