package io.metaloom.jdlib.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class Image {

	public final int width;
	public final int height;
	private final boolean hasAlphaChannel;
	private int pixelLength;
	public final byte[] pixels;

	public Image(BufferedImage image) {
		DataBuffer buffer = image.getRaster().getDataBuffer();
		if (buffer instanceof DataBufferByte) {
			this.pixels = ((DataBufferByte) buffer).getData();
		} else if (buffer instanceof DataBufferInt) {
			int[] data = ((DataBufferInt) buffer).getData();
			ByteBuffer byteBuffer = ByteBuffer.allocate(data.length * 4);
			IntBuffer intBuffer = byteBuffer.asIntBuffer();
			intBuffer.put(data);
			this.pixels = byteBuffer.array();
		} else {
			throw new RuntimeException("Encountered unknown buffer type " + buffer.getClass().getSimpleName());
		}
		this.width = image.getWidth();
		this.height = image.getHeight();
		this.hasAlphaChannel = image.getAlphaRaster() != null;
		this.pixelLength = 3;
		if (hasAlphaChannel) {
			this.pixelLength = 4;
		}
	}

	short[] getRGB(int x, int y) {
		int pos = (y * pixelLength * width) + (x * pixelLength);
		short rgb[] = new short[4];
		if (hasAlphaChannel) {
			rgb[3] = (short) (pixels[pos++] & 0xFF); // Alpha
		}
		rgb[2] = (short) (pixels[pos++] & 0xFF); // Blue
		rgb[1] = (short) (pixels[pos++] & 0xFF); // Green
		rgb[0] = (short) (pixels[pos++] & 0xFF); // Red
		return rgb;
	}
}
