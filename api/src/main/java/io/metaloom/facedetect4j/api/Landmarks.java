package io.metaloom.facedetect4j.api;

/**
 * The five facial keypoints, in <b>ArcFace order</b> and absolute source-image pixels:
 *
 * <pre>
 *   0  left eye           (image-left, i.e. the subject's right)
 *   1  right eye
 *   2  nose tip
 *   3  left mouth corner
 *   4  right mouth corner
 * </pre>
 *
 * This order is the interop contract for alignment, not a naming convention. Permuting it produces
 * a crop that still looks like a face and still embeds without error — it just scores worse, with
 * nothing in the pipeline reporting a problem.
 */
public record Landmarks(float[] xy) {

	public Landmarks {
		if (xy.length != 10) {
			throw new IllegalArgumentException("expected 5 points (10 floats), got " + xy.length);
		}
	}

	public float x(int i) {
		return xy[2 * i];
	}

	public float y(int i) {
		return xy[2 * i + 1];
	}

	public Landmarks scaled(float f) {
		float[] o = new float[10];
		for (int i = 0; i < 10; i++) {
			o[i] = xy[i] * f;
		}
		return new Landmarks(o);
	}

	/**
	 * Cheap plausibility check on the ordering: image-left eye must sit left of the right eye, and
	 * the eyes must sit above the mouth. Useful as an assertion when integrating a new detector.
	 */
	public boolean geometryLooksSane() {
		float eyeY = (y(0) + y(1)) / 2f;
		float mouthY = (y(3) + y(4)) / 2f;
		return x(0) < x(1) && eyeY < mouthY;
	}
}
