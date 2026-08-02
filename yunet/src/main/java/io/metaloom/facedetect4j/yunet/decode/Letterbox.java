package io.metaloom.facedetect4j.yunet.decode;

import io.metaloom.facedetect4j.api.FaceImage;

/**
 * Blob construction for models that take raw BGR.
 *
 * <h2>YuNet and SFace take pixels as-is</h2>
 * OpenCV drives both with a bare {@code blobFromImage(img)} — scale 1.0, no mean subtraction, no
 * channel swap. That is unusual: the ArcFace family wants {@code (RGB - 127.5) / 127.5} and SCRFD
 * wants {@code (RGB - 127.5) / 128}. Applying either of those normalisations here produces
 * embeddings that are stable, comparable to each other, and quietly much worse.
 */
public final class Letterbox {

	private Letterbox() {
	}

	/** Scale factor that fits the image inside {@code netW x netH} without distorting it. */
	public static float scaleFor(int imgW, int imgH, int netW, int netH) {
		return Math.min((float) netW / imgW, (float) netH / imgH);
	}

	/**
	 * Aspect-preserving resize into a zero-padded {@code netW x netH} canvas, NCHW float32, raw
	 * BGR values.
	 */
	public static float[] blob(FaceImage img, int netW, int netH, float scale) {
		float[] out = new float[3 * netH * netW];
		int plane = netH * netW;
		int newW = Math.min(netW, Math.max(1, Math.round(img.width() * scale)));
		int newH = Math.min(netH, Math.max(1, Math.round(img.height() * scale)));

		for (int y = 0; y < newH; y++) {
			double sy = (y + 0.5) / scale - 0.5;
			int y0 = (int) Math.floor(sy);
			double fy = sy - y0;
			for (int x = 0; x < newW; x++) {
				double sx = (x + 0.5) / scale - 0.5;
				int x0 = (int) Math.floor(sx);
				double fx = sx - x0;
				for (int c = 0; c < 3; c++) {
					out[c * plane + y * netW + x] = (float) bilinear(img, x0, y0, fx, fy, c);
				}
			}
		}
		return out;
	}

	/** 112x112 aligned crop to NCHW float32, raw BGR. */
	public static float[] cropBlob(byte[] bgr112, int size) {
		float[] out = new float[3 * size * size];
		int plane = size * size;
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) {
				int o = (y * size + x) * 3;
				for (int c = 0; c < 3; c++) {
					out[c * plane + y * size + x] = bgr112[o + c] & 0xFF;
				}
			}
		}
		return out;
	}

	private static double bilinear(FaceImage img, int x0, int y0, double fx, double fy, int c) {
		int w = img.width(), h = img.height();
		int cx0 = clamp(x0, 0, w - 1), cy0 = clamp(y0, 0, h - 1);
		int cx1 = clamp(x0 + 1, 0, w - 1), cy1 = clamp(y0 + 1, 0, h - 1);
		double v00 = img.at(cx0, cy0, c), v10 = img.at(cx1, cy0, c);
		double v01 = img.at(cx0, cy1, c), v11 = img.at(cx1, cy1, c);
		double top = v00 + (v10 - v00) * fx;
		double bot = v01 + (v11 - v01) * fx;
		return top + (bot - top) * fy;
	}

	private static int clamp(int v, int lo, int hi) {
		return v < lo ? lo : (v > hi ? hi : v);
	}
}
