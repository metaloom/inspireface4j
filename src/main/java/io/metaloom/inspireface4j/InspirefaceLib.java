package io.metaloom.inspireface4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.metaloom.inspireface4j.data.HFFaceAttributeResult;
import io.metaloom.inspireface4j.data.HFFaceFeature;
import io.metaloom.inspireface4j.data.HFLogLevel;
import io.metaloom.inspireface4j.data.HFMultipleFaceData;
import io.metaloom.inspireface4j.data.HFaceRect;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;

public class InspirefaceLib {

	private static final Logger logger = LoggerFactory.getLogger(InspirefaceLib.class);

	static final Linker linker = Linker.nativeLinker();
	static final Arena arena = Arena.ofAuto();

	static final AddressLayout ADDR = ValueLayout.ADDRESS.withOrder(ByteOrder.nativeOrder());
	static final OfInt JINT = ValueLayout.JAVA_INT.withOrder(ByteOrder.nativeOrder());
	static final OfLong JSIZE_T = ValueLayout.JAVA_LONG.withOrder(ByteOrder.nativeOrder()); // size_t on 64-bit Linux

	private static SymbolLookup inspirefaceLibrary;

	private static boolean initialized = false;

	private static MemorySegment globalMultipleFaceData;

	private static void checkInitialized() {
		if (!initialized) {
			throw new RuntimeException("InspirefaceLib not initialized");
		}
	}

	private static List<Detection> mapFaceDetections(MemorySegment multipleFaceData) {
		List<Detection> detections = new ArrayList<>();
		HFMultipleFaceData faceData = new HFMultipleFaceData(multipleFaceData);
		multipleFaceData = faceData.segment();
		int detectedNum = multipleFaceData.get(ValueLayout.JAVA_INT, 0);

		MemorySegment rectsPointer = multipleFaceData.get(ValueLayout.ADDRESS,
			HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("rects"))); // you need to calculate this

		MemorySegment confPointer = multipleFaceData.get(ValueLayout.ADDRESS,
			HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("rects"))); // you need to calculate this
		// offset

		MemorySegment confArray = MemorySegment.ofAddress(confPointer.address());
		confArray = confArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		MemorySegment rectsArray = MemorySegment.ofAddress(rectsPointer.address());
		rectsArray = rectsArray.reinterpret(detectedNum * HFaceRect.FACE_RECT_LAYOUT.byteSize());
		for (int i = 0; i < detectedNum; i++) {

			MemorySegment rect = rectsArray.asSlice(i * HFaceRect.FACE_RECT_LAYOUT.byteSize(), HFaceRect.FACE_RECT_LAYOUT.byteSize());

			float conf = confArray.get(ValueLayout.JAVA_FLOAT, 0);
			int x = (int) HFaceRect.X_HANDLE.get(rect, 0);
			int y = (int) HFaceRect.Y_HANDLE.get(rect, 0);
			int width = (int) HFaceRect.WIDTH_HANDLE.get(rect, 0);
			int height = (int) HFaceRect.HEIGHT_HANDLE.get(rect, 0);

			detections.add(new Detection(new BoundingBox(x, y, width, height), conf));
			// System.out.printf("Face[%d] - x: %d, y: %d, width: %d, height: %d - Conf: %.2f %n", i, x, y, width, height, conf);
		}

		return detections;
	}

	private static SymbolLookup loadLib() {
		String os = System.getProperty("os.name", "generic")
			.toLowerCase(Locale.ENGLISH);
		String name = System.mapLibraryName("jinspireface");

		String libpath = "";
		if (os.contains("linux")) {
			libpath = "/native" + File.separator + "linux" + File.separator + name;
		} else if (os.contains("mac")) {
			libpath = "/native" + File.separator + "macosx" + File.separator + name;
		} else {
			throw new java.lang.UnsupportedOperationException(os + " is not supported. Try to recompile Jdlib on your machine and then use it.");
		}

		try (InputStream inputStream = InspirefaceLib.class.getResourceAsStream(libpath)) {
			File fileOut = File.createTempFile(name, "");
			try (OutputStream outputStream = new FileOutputStream(fileOut)) {
				byte[] buffer = new byte[1024];
				int read = -1;
				while ((read = inputStream.read(buffer)) != -1) {
					outputStream.write(buffer, 0, read);
				}
				System.load(fileOut.toString());
			}
			return SymbolLookup.loaderLookup();
			// inspirefaceLib = SymbolLookup.libraryLookup(libPath, arena);
		} catch (Exception e) {
			throw new RuntimeException("Failed to load native library.", e);
		}
	}

	/**
	 * 
	 * @param modelPath
	 * @param detectPixelLevel Usually 160, 192, 256, 320, 640
	 */
	public static void init(String modelPath, int detectPixelLevel) {
		if (initialized) {
			throw new RuntimeException("InspirefaceLib already initialized");
		}
		inspirefaceLibrary = loadLib();

		Path mPath = Paths.get(modelPath);
		if (!Files.exists(mPath)) {
			throw new RuntimeException("Unable to locate model with path " + mPath);
		}

		MethodHandle initHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("initializeSession"),
				FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

		try (Arena arena = Arena.ofConfined()) {
			// MemorySegment labelsPathMem = arena.allocateFrom(labelsPath);
			MemorySegment modelPathMem = arena.allocateFrom(modelPath);
			initHandler.invoke(modelPathMem, detectPixelLevel);
			initialized = true;
		} catch (Throwable t) {
			throw new RuntimeException("Failed to initialize InspirefaceLib", t);
		}
	}

	public static List<Detection> detect(Mat imageMat, boolean drawBoundingBoxes) {
		checkInitialized();

		MethodHandle detectHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("detect"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

		try {
			MemorySegment imageSeg = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment multipleFaceData = (MemorySegment) detectHandler.invoke(imageSeg, drawBoundingBoxes);
			globalMultipleFaceData = multipleFaceData.reinterpret(HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteSize());
			return mapFaceDetections(multipleFaceData);
			// return new ArrayList<Detection>();
		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke detection", t);
		}

	}

	public static List<Detection> detect(BufferedImage img, boolean drawBoundingBoxes) {
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		return detect(imageMat, drawBoundingBoxes);
	}

	public static List<FaceAttributes> attributes(Mat imageMat, boolean drawAttributes) {
		checkInitialized();

		MethodHandle attrHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceAttributes"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

		if (globalMultipleFaceData == null) {
			throw new RuntimeException("Data is null - run detect first");
		}
		try {
			MemorySegment imageAttr = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment attrData = (MemorySegment) attrHandler.invoke(globalMultipleFaceData, imageAttr);

			HFFaceAttributeResult attr = new HFFaceAttributeResult(attrData);
			List<FaceAttributes> attributes = new ArrayList<>();
			for (int i = 0; i < attr.numFaces(); i++) {
				attributes.add(new FaceAttributes(attr.race(i), attr.gender(i), attr.age(i)));
//				System.out.println("ATTR_NUM[" + i + "]: " + attr.numFaces());
//				System.out.println("ATTR_RACE[" + i + "]: " + attr.race(i));
//				System.out.println("ATTR_GENDER[" + i + "]: " + attr.gender(i));
//				System.out.println("ATTR_AGE[" + i + "]: " + attr.age(i));
			}
			return attributes;
			// return mapFaceDetections(multipleFaceData);

		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke attributes", t);
		}

	}

	public static float[] embedding(Mat imageMat, int faceNr) {
		checkInitialized();

		MethodHandle attrHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceEmbeddings"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

		if (globalMultipleFaceData == null) {
			throw new RuntimeException("Data is null - run detect first");
		}
		try {
			MemorySegment imageAttr = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment embeddingData = (MemorySegment) attrHandler.invoke(globalMultipleFaceData, imageAttr, faceNr);
			HFFaceFeature attr = new HFFaceFeature(embeddingData);
			return attr.data();
		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke embeddings", t);
		}
	}

	public static void logLevel(HFLogLevel level) {
		MethodHandle levelHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("logLevel"),
				FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
		try {
			levelHandler.invoke(level.ordinal());
		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke embeddings", t);
		}
	}
}
