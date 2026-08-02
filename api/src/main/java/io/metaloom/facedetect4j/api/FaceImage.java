package io.metaloom.facedetect4j.api;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import javax.imageio.ImageIO;

/**
 * An image in <b>BGR, HWC, 8-bit, row-major</b> layout, stride {@code width * 3}.
 *
 * <p>
 * BGR because every model this library drives was exported from an OpenCV-shaped pipeline and
 * consumes BGR directly. Standardising on it means exactly one conversion site instead of one at
 * every boundary.
 *
 * <p>
 * Deliberately a plain byte array rather than an OpenCV {@code Mat}: that keeps this library free
 * of any native image dependency, so a caller already holding OpenCV 4.x or 5.x natives (or none)
 * can use it without a version conflict. {@link #ofBufferedImage} covers AWT callers and
 * {@link #ofBgrBytes} covers callers who already have raw frame bytes — a video decoder, say.
 */
public record FaceImage(int width, int height, byte[] bgr) {

	public FaceImage {
		int expect = width * height * 3;
		if (bgr.length != expect) {
			throw new IllegalArgumentException(
				"bgr length " + bgr.length + " != width*height*3 (" + expect + ")");
		}
	}

	/** Wrap an existing BGR buffer without copying. The caller must not mutate it afterwards. */
	public static FaceImage ofBgrBytes(int width, int height, byte[] bgr) {
		return new FaceImage(width, height, bgr);
	}

	public static FaceImage read(Path path) throws IOException {
		BufferedImage img = ImageIO.read(path.toFile());
		if (img == null) {
			throw new IOException("no ImageIO reader for " + path);
		}
		return ofBufferedImage(img);
	}

	public static FaceImage read(File file) throws IOException {
		return read(file.toPath());
	}

	public static FaceImage ofBufferedImage(BufferedImage src) {
		int w = src.getWidth();
		int h = src.getHeight();
		// Fast path: already the layout we want, so this is a straight array copy.
		if (src.getType() == BufferedImage.TYPE_3BYTE_BGR) {
			byte[] raw = ((DataBufferByte) src.getRaster().getDataBuffer()).getData();
			return new FaceImage(w, h, raw.clone());
		}
		byte[] bgr = new byte[w * h * 3];
		int[] row = new int[w];
		for (int y = 0; y < h; y++) {
			src.getRGB(0, y, w, 1, row, 0, w);
			int o = y * w * 3;
			for (int x = 0; x < w; x++) {
				int argb = row[x];
				bgr[o++] = (byte) (argb & 0xFF);
				bgr[o++] = (byte) ((argb >> 8) & 0xFF);
				bgr[o++] = (byte) ((argb >> 16) & 0xFF);
			}
		}
		return new FaceImage(w, h, bgr);
	}

	/** Unsigned channel value. {@code c}: 0 = B, 1 = G, 2 = R. */
	public int at(int x, int y, int c) {
		return bgr[(y * width + x) * 3 + c] & 0xFF;
	}
}
