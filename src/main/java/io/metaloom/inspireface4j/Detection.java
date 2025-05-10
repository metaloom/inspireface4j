package io.metaloom.inspireface4j;

public record Detection(BoundingBox box, float conf, int classId) {

	public String label() {
		return InspirefaceLib.toLabel(this);
	}

}
