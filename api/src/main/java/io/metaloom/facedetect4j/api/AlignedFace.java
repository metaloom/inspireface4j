package io.metaloom.facedetect4j.api;

/**
 * A 112x112 BGR crop produced by the canonical ArcFace five-point alignment, plus the 2x3 affine
 * that produced it (row-major {@code [a,b,tx,c,d,ty]}).
 *
 * <p>
 * Exposed in the API rather than hidden inside the embedder so callers can cache crops, embed the
 * same crop with more than one model, or feed a crop they aligned themselves.
 */
public record AlignedFace(byte[] bgr, float[] affine2x3, Face source) {

	public static final int SIZE = 112;

	public AlignedFace {
		if (bgr.length != SIZE * SIZE * 3) {
			throw new IllegalArgumentException(
				"crop must be " + SIZE + "x" + SIZE + "x3 bytes, got " + bgr.length);
		}
		if (affine2x3.length != 6) {
			throw new IllegalArgumentException("affine must hold 6 floats, got " + affine2x3.length);
		}
	}

	public int at(int x, int y, int c) {
		return bgr[(y * SIZE + x) * 3 + c] & 0xFF;
	}

	/** Standard deviation over all channels. Near zero means the warp sampled outside the image. */
	public double stddev() {
		double sum = 0, sq = 0;
		for (byte b : bgr) {
			int v = b & 0xFF;
			sum += v;
			sq += (double) v * v;
		}
		double n = bgr.length;
		double mean = sum / n;
		return Math.sqrt(Math.max(0, sq / n - mean * mean));
	}
}
