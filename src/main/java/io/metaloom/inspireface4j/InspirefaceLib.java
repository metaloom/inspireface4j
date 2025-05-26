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

	// private static List<Detection> mapDetectionsArray(MemorySegment detectionArrayStruct) throws Throwable {
	//
	// MethodHandle freeDetectionHandler = linker.downcallHandle(
	// inspirefaceLibrary.find("free_detection").orElseThrow(),
	// FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
	//
	// detectionArrayStruct = detectionArrayStruct.reinterpret(HFMultipleFaceDataLayout.size());
	//
	// System.out.println("--------------");
	// int d2 = HFMultipleFaceDataLayout.getDetectedNums(detectionArrayStruct);
	// System.out.println("d2: " + d2);
	// MemorySegment dataPtr = detectionArrayStruct.get(ValueLayout.ADDRESS, 0);
	// int detectionCount = detectionArrayStruct.get(ValueLayout.JAVA_INT, ValueLayout.ADDRESS.byteSize());
	// dataPtr = dataPtr.reinterpret(DetectionMemoryLayout.size() * detectionCount);
	//
	// if (logger.isDebugEnabled()) {
	// logger.debug("Received " + detectionCount + " detections");
	// logger.debug("Got " + detectionArrayStruct + " from function.");
	// }
	//
	// // Read BoundingBox elements
	// List<Detection> detections = new ArrayList<>();
	// long structSize = DetectionMemoryLayout.size();
	// for (long i = 0; i < detectionCount; i++) {
	// MemorySegment detectionMemory = dataPtr.asSlice(i * structSize, structSize);
	// float conf = DetectionMemoryLayout.getConf(detectionMemory);
	// int clazz = DetectionMemoryLayout.getClassId(detectionMemory);
	// int x = detectionMemory.get(JINT, 0);
	// int y = detectionMemory.get(JINT, JINT.byteSize());
	// int width = detectionMemory.get(JINT, 2 * JINT.byteSize());
	// int height = detectionMemory.get(JINT, 3 * JINT.byteSize());
	// BoundingBox bbox = new BoundingBox(x, y, width, height);
	// detections.add(new Detection(bbox, conf, clazz));
	// }
	//
	// // Print results
	// // detections.forEach(System.out::println);
	//
	// // Free the native memory
	// freeDetectionHandler.invoke(detectionArrayStruct);
	// return detections;
	//
	// }

	private static List<Detection> mapFaceDetections(MemorySegment multipleFaceData) {
		List<Detection> detections = new ArrayList<>();

		HFMultipleFaceData faceData = new HFMultipleFaceData(multipleFaceData);
		// System.out.println("Detections: " + faceData.detectedNum());
		// if (faceData.detectedNum() >= 1) {
		// MemorySegment rectData = faceData.rectsMemory();
		// HFaceRect rect = new HFaceRect(rectData, 0);
		// System.out.println(rect);
		// }

		// System.out.println("---------");
		// // System.out.println("SIZE: " + HFMultipleFaceDataLayout.size());
		// multipleFaceData = multipleFaceData.reinterpret(HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteSize());
		multipleFaceData = faceData.segment();
		//
		//
		//
		int detectedNum = multipleFaceData.get(ValueLayout.JAVA_INT, 0);
		//
		//
		////		VarHandle DETECTED_NUM = HFMultipleFaceDataLayout.DETECTION_ARRAY_LAYOUT.varHandle(
////			MemoryLayout.PathElement.groupElement("detectedNum"));
////		int numFaces = (int) DETECTED_NUM.get(multipleFaceData, 0L);
////		System.out.println("NUM2: " + numFaces);
		//
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
			System.out.printf("Face[%d] - x: %d, y: %d, width: %d, height: %d - Conf: %.2f %n", i, x, y, width, height, conf);
		}

		// // Rects
		// SequenceLayout pointsLayout = MemoryLayout.sequenceLayout(detectedNum, HFMultipleFaceDataLayout.FACE_RECT_LAYOUT);
		// VarHandle xHandle = pointsLayout.varHandle(MemoryLayout.PathElement.sequenceElement(),
		// MemoryLayout.PathElement.groupElement("x"));

		// for (int i = 0; i < detectedNum; i++) {
		// System.out.println("X[" + i +"] " + xHandle.get(multipleFaceData, 0, i));
		// }
		// MemorySegment rect = HFMultipleFaceDataLayout.RECTS_HANDLER..get(multipleFaceData, 0L);

		// System.out.println("Code: " + code);
		// List<Detection> results = mapDetectionsArray(detectionArrayStruct);
		// return results;

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

	public static void init(String modelPath) {
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
				FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

		try (Arena arena = Arena.ofConfined()) {
			// MemorySegment labelsPathMem = arena.allocateFrom(labelsPath);
			MemorySegment modelPathMem = arena.allocateFrom(modelPath);
			initHandler.invoke(modelPathMem);
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
			// return mapFaceDetections(multipleFaceData);
			return new ArrayList<Detection>();
		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke detection", t);
		}

	}

	public static List<Detection> detect(BufferedImage img, boolean drawBoundingBoxes) {
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		return detect(imageMat, drawBoundingBoxes);
	}

	public static void attributes(Mat imageMat, boolean drawAttributes) {
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

			System.out.println(attrData);
			HFFaceAttributeResult attr = new HFFaceAttributeResult(attrData);
			for (int i = 0; i < attr.numFaces(); i++) {
				System.out.println("ATTR_NUM[" + i + "]: " + attr.numFaces());
				System.out.println("ATTR_RACE[" + i + "]: " + attr.race(i));
				System.out.println("ATTR_GENDER[" + i + "]: " + attr.gender(i));
				System.out.println("ATTR_AGE[" + i + "]: " + attr.age(i));
			}
			// return mapFaceDetections(multipleFaceData);

		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke attributes", t);
		}

	}

	public static void embedding(Mat imageMat) {
		checkInitialized();

		MethodHandle attrHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceEmbeddings"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

		if (globalMultipleFaceData == null) {
			throw new RuntimeException("Data is null - run detect first");
		}
		try {
			MemorySegment imageAttr = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment embeddingData = (MemorySegment) attrHandler.invoke(globalMultipleFaceData, imageAttr);

			System.out.println(embeddingData);
			HFFaceFeature attr = new HFFaceFeature(embeddingData);
			System.out.println("Embedding size: " + attr.size());
			
			attr.data();

		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke embeddings", t);
		}
	}
}
