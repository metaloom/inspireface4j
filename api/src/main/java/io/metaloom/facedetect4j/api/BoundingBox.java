package io.metaloom.facedetect4j.api;

/** Axis-aligned box in absolute pixel coordinates of the source image. */
public record BoundingBox(float x1, float y1, float x2, float y2) {

	public static BoundingBox ofXYWH(float x, float y, float w, float h) {
		return new BoundingBox(x, y, x + w, y + h);
	}

	public float width() {
		return x2 - x1;
	}

	public float height() {
		return y2 - y1;
	}

	public float area() {
		return Math.max(0, width()) * Math.max(0, height());
	}

	public float centerX() {
		return (x1 + x2) * 0.5f;
	}

	public float centerY() {
		return (y1 + y2) * 0.5f;
	}

	public BoundingBox scaled(float f) {
		return new BoundingBox(x1 * f, y1 * f, x2 * f, y2 * f);
	}

	/** Intersection over union, standard convention (no pixel offset). */
	public double iou(BoundingBox o) {
		double iw = Math.min(x2, o.x2) - Math.max(x1, o.x1);
		double ih = Math.min(y2, o.y2) - Math.max(y1, o.y1);
		if (iw <= 0 || ih <= 0) {
			return 0;
		}
		double inter = iw * ih;
		double union = area() + o.area() - inter;
		return union <= 0 ? 0 : inter / union;
	}
}
