package io.metaloom.jdlib;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.metaloom.jdlib.util.FaceDescriptor;
import io.metaloom.jdlib.util.Image;

public class Jdlib {

	private static final Logger log = LoggerFactory.getLogger(Jdlib.class);

	private String facialLandmarksModelPath;
	private String faceEmbeddingModelPath;
	private String cnnFaceDetectorModelPath;

	public Jdlib(String facialLandmarksModelPath, String faceEmbeddingModelPath, String cnnFaceDetectorModelPath) {
		this.facialLandmarksModelPath = facialLandmarksModelPath;
		this.faceEmbeddingModelPath = faceEmbeddingModelPath;
		this.cnnFaceDetectorModelPath = cnnFaceDetectorModelPath;
		loadLib();
	}

	public Jdlib(String facialLandmarksModelPath) {
		this.facialLandmarksModelPath = facialLandmarksModelPath;
		this.faceEmbeddingModelPath = null;
		this.cnnFaceDetectorModelPath = null;
		loadLib();
	}

	private native long getFaceDetectorHandler();

	private native long getShapePredictorHandler(String modelPath);

	private native long getFaceEmbeddingHandler(String modelPath);

	private native long getCNNFaceDetectorHandler(String modelPath);

	private native List<Rectangle> faceDetect(long faceDetectorHandler, byte[] pixels, int h, int w);

	private native List<Rectangle> cnnFaceDetect(long cnnFaceDetectorHandler, byte[] pixels, int h, int w);

	private native List<FaceDescriptor> getFacialLandmarks(long shapePredictorHandler, long faceDetectorHandler, byte[] pixels, int h, int w);

	private native List<FaceDescriptor> getFaceEmbeddings(long FaceEmbeddingHandler, long shapePredictorHandler, long faceDetectorHandler,
			byte[] pixels, int h, int w);

	private void loadLib() {
		String os = System.getProperty("os.name", "generic")
				.toLowerCase(Locale.ENGLISH);
		String name = System.mapLibraryName("jdlib");

		String libpath = "";
		if (os.contains("linux")) {
			libpath = "/native" + File.separator + "linux" + File.separator + name;
		} else if (os.contains("mac")) {
			libpath = "/native" + File.separator + "macosx" + File.separator + name;
		} else {
			throw new java.lang.UnsupportedOperationException(os + " is not supported. Try to recompile Jdlib on your machine and then use it.");
		}

		try (InputStream inputStream = Jdlib.class.getResourceAsStream(libpath)) {
			File fileOut = File.createTempFile(name, "");
			try (OutputStream outputStream = new FileOutputStream(fileOut)) {
				byte[] buffer = new byte[1024];
				int read = -1;
				while ((read = inputStream.read(buffer)) != -1) {
					outputStream.write(buffer, 0, read);
				}
				System.load(fileOut.toString());
			}
		} catch (Exception e) {
			log.error("Failed to load jdlib native library.", e);
		}
	}

	public List<Rectangle> detectFace(BufferedImage img) {
		Image image = new Image(img);
		List<Rectangle> data = faceDetect(getFaceDetectorHandler(), image.pixels, image.height, image.width);
		if (data == null) {
			log.error("Jdlib | detectFace | Null data!!");
			data = new ArrayList<>();
		}
		return data;
	}

	public List<FaceDescriptor> getFaceLandmarks(BufferedImage img) {
		Image image = new Image(img);
		List<FaceDescriptor> data = getFacialLandmarks(getShapePredictorHandler(facialLandmarksModelPath), getFaceDetectorHandler(), image.pixels,
				image.height, image.width);
		if (data == null) {
			log.error("Jdlib | getFaceLandmarks | Null data!");
			data = new ArrayList<>();
		}
		return data;
	}

	public List<FaceDescriptor> getFaceEmbeddings(BufferedImage img) {
		if (facialLandmarksModelPath == null) {
			throw new IllegalArgumentException("Path to face embedding model isn't provided!");
		}

		Image image = new Image(img);
		List<FaceDescriptor> data = getFaceEmbeddings(getFaceEmbeddingHandler(faceEmbeddingModelPath),
				getShapePredictorHandler(facialLandmarksModelPath), getFaceDetectorHandler(), image.pixels, image.height, image.width);
		if (data == null) {
			log.error("Jdlib | getFaceEmbeddings | Null data!");
			data = new ArrayList<>();
		}
		return data;
	}

	public List<Rectangle> cnnDetectFace(BufferedImage img) {
		if (cnnFaceDetectorModelPath == null) {
			throw new IllegalArgumentException("Path to cnn face dector model isn't provided!");
		}
		Image image = new Image(img);
		List<Rectangle> data = cnnFaceDetect(getCNNFaceDetectorHandler(cnnFaceDetectorModelPath), image.pixels, image.height, image.width);
		if (data == null) {
			log.error("Jdlib | cnnDetectFace | Null data!");
			data = new ArrayList<>();
		}
		return data;
	}
}
