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
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.inspireface4j.data.internal.HFFaceAttributeResult;
import io.metaloom.inspireface4j.data.internal.HFFaceFeature;
import io.metaloom.inspireface4j.data.internal.HFLogLevel;
import io.metaloom.inspireface4j.data.internal.HFMultipleFaceData;
import io.metaloom.inspireface4j.data.internal.HFaceRect;
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

	// private static MemorySegment globalMultipleFaceData;

	private static void checkInitialized() {
		if (!initialized) {
			throw new Inspireface4jException("InspirefaceLib not initialized");
		}
	}

	private static FaceDetections mapFaceDetections(MemorySegment multipleFaceData) {
		HFMultipleFaceData faceData = new HFMultipleFaceData(multipleFaceData);
		FaceDetections detections = new FaceDetections(faceData);
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
			logger.error("Failed to load inspireface4j lib", e);
			throw new Inspireface4jException("Failed to load native library.", e);
		}
	}

	public static InspirefaceSession session(String modelPath, int detectPixelLevel) {
		inspirefaceLibrary = loadLib();

		Path mPath = Paths.get(modelPath);
		if (!Files.exists(mPath)) {
			throw new Inspireface4jException("Unable to locate model with path " + mPath);
		}

		MethodHandle createHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("createSession"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

		try (Arena arena = Arena.ofConfined()) {
			// MemorySegment labelsPathMem = arena.allocateFrom(labelsPath);
			MemorySegment modelPathMem = arena.allocateFrom(modelPath);
			MemorySegment sessionPtr = (MemorySegment) createHandler.invoke(modelPathMem, detectPixelLevel);
			initialized = true;
			return new InspirefaceSession(sessionPtr);
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to initialize InspirefaceLib", t);
		}

	}

	/**
	 * 
	 * @param modelPath
	 * @param detectPixelLevel
	 *            Usually 160, 192, 256, 320, 640
	 */
	public static void init(String modelPath, int detectPixelLevel) {
		if (initialized) {
			throw new Inspireface4jException("InspirefaceLib already initialized");
		}
		inspirefaceLibrary = loadLib();

		Path mPath = Paths.get(modelPath);
		if (!Files.exists(mPath)) {
			throw new Inspireface4jException("Unable to locate model with path " + mPath);
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
			throw new Inspireface4jException("Failed to initialize InspirefaceLib", t);
		}
	}

	public static FaceDetections detect(InspirefaceSession session, Mat imageMat, boolean drawBoundingBoxes) {
		checkInitialized();

		MethodHandle detectHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("detect"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

		try {
			MemorySegment imageSeg = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment multipleFaceData = (MemorySegment) detectHandler.invoke(session.data(), imageSeg, drawBoundingBoxes);
			multipleFaceData = multipleFaceData.reinterpret(HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteSize());
			return mapFaceDetections(multipleFaceData);
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to invoke detection", t);
		}

	}

	public static FaceDetections detect(InspirefaceSession session, BufferedImage img) {
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		FaceDetections detections = detect(session, imageMat, false);
		MatProvider.released(imageMat);
		return detections;
	}

	public static List<FaceAttributes> attributes(InspirefaceSession session, Mat imageMat, FaceDetections detections, boolean drawAttributes) {
		checkInitialized();

		MethodHandle attrHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceAttributes"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

		try {
			MemorySegment imageAttr = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment attrData = (MemorySegment) attrHandler.invoke(session.data(), detections.data().segment(), imageAttr);

			HFFaceAttributeResult attr = new HFFaceAttributeResult(attrData);
			List<FaceAttributes> attributes = new ArrayList<>();
			for (int i = 0; i < attr.numFaces(); i++) {

				attributes.add(new FaceAttributes(attr.race(i), attr.gender(i), attr.age(i)));
				if (drawAttributes) {
					Detection detection = detections.get(i);
					BoundingBox box = detection.box();
					Point p = new Point(box.x, box.y - 5);
					Scalar color = new Scalar(200f, 200f, 0f);
					float fontWeight = 0.8f;
					int thickness = 1;
					int yStep = 12;
					CVUtils.drawText(imageMat, "Race: " + attr.race(i).name(), p, fontWeight, color, thickness);
					p.y = p.y - yStep;
					CVUtils.drawText(imageMat, "Gender: " + attr.gender(i).name(), p, fontWeight, color, thickness);
					p.y = p.y - yStep;
					CVUtils.drawText(imageMat, "Age: " + attr.age(i).name(), p, fontWeight, color, thickness);
				}
			}
			return attributes;
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to invoke attributes", t);
		}

	}

	public static void releaseFaceFeature(HFMultipleFaceData feature) {
		MethodHandle methodHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("releaseFaceFeature"),
				FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

		try {
			methodHandler.invoke(feature.segment());
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to release native face data", t);
		}
	}

	public static float[] embedding(InspirefaceSession session, Mat imageMat, FaceDetections detections, int faceNr) {
		checkInitialized();

		MethodHandle attrHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceEmbeddings"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));

		try {
			MemorySegment imageAttr = MemorySegment.ofAddress(imageMat.getNativeObjAddr());
			MemorySegment embeddingData = (MemorySegment) attrHandler.invoke(session.data(), detections.data().segment(), imageAttr, faceNr);
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
			throw new Inspireface4jException("Failed to invoke embeddings", t);
		}
	}

	public static void releaseSession(MemorySegment sessionPtr) throws Inspireface4jException {
		MethodHandle releaseSessionHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("releaseSession"),
				FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

		try {
			releaseSessionHandler.invoke(sessionPtr);
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to release session", t);
		}
	}
}
