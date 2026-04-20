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

import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.core.Point;
import io.metaloom.opencv.core.Scalar;
import io.metaloom.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.inspireface4j.data.internal.HFFaceAttributeResult;
import io.metaloom.inspireface4j.data.internal.HFFaceEulerAngle;
import io.metaloom.inspireface4j.data.internal.HFFaceFeature;
import io.metaloom.inspireface4j.data.internal.HFLandmarkData;
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
			HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("rects")));

		MemorySegment confPointer = multipleFaceData.get(ValueLayout.ADDRESS,
			HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("detConfidence")));

		MemorySegment confArray = MemorySegment.ofAddress(confPointer.address());
		confArray = confArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		MemorySegment rectsArray = MemorySegment.ofAddress(rectsPointer.address());
		rectsArray = rectsArray.reinterpret(detectedNum * HFaceRect.FACE_RECT_LAYOUT.byteSize());

		// Angular Data

		long offsetAngles = 32; // After detConfidence
		MemorySegment anglesStruct = multipleFaceData.asSlice(offsetAngles);
		anglesStruct = anglesStruct.reinterpret(3 * ValueLayout.ADDRESS.byteSize());

		// Step 3: Extract pointers to float arrays
		// MemorySegment rollPtr = anglesStruct.get(ValueLayout.ADDRESS, 0);

		MemorySegment rollPointer = anglesStruct.get(ValueLayout.ADDRESS,
			HFFaceEulerAngle.HFFACE_EULER_ANGLE_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("roll")));
		MemorySegment yawPointer = anglesStruct.get(ValueLayout.ADDRESS,
			HFFaceEulerAngle.HFFACE_EULER_ANGLE_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("yaw")));
		MemorySegment pitchPointer = anglesStruct.get(ValueLayout.ADDRESS,
			HFFaceEulerAngle.HFFACE_EULER_ANGLE_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("pitch")));

		// Step 4: Interpret pointers as segments of floats
		// Assuming face count equals number of floats in each array
		MemorySegment rollArray = MemorySegment.ofAddress(rollPointer.address());
		rollArray = rollArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		MemorySegment yawArray = MemorySegment.ofAddress(yawPointer.address());
		yawArray = yawArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		MemorySegment pitchArray = MemorySegment.ofAddress(pitchPointer.address());
		pitchArray = pitchArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		//
		// MemorySegment angelData = multipleFaceData.get(ValueLayout.ADDRESS,
		// HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("angels")));
		//
		// MemorySegment yawArray = MemorySegment.ofAddress(angelData.address());
		// yawArray = yawArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);
		//
		// MemorySegment rollArray = MemorySegment.ofAddress(angelData.address());
		// rollArray = rollArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);
		//
		// MemorySegment pitchArray = MemorySegment.ofAddress(angelData.address());
		// pitchArray = pitchArray.reinterpret(ValueLayout.JAVA_FLOAT.byteSize() * detectedNum);

		for (int i = 0; i < detectedNum; i++) {

			MemorySegment rect = rectsArray.asSlice(i * HFaceRect.FACE_RECT_LAYOUT.byteSize(), HFaceRect.FACE_RECT_LAYOUT.byteSize());

			// TODO fix conf
			float conf = confArray.get(ValueLayout.JAVA_FLOAT, 0);
			int x = (int) HFaceRect.X_HANDLE.get(rect, 0);
			int y = (int) HFaceRect.Y_HANDLE.get(rect, 0);
			int width = (int) HFaceRect.WIDTH_HANDLE.get(rect, 0);
			int height = (int) HFaceRect.HEIGHT_HANDLE.get(rect, 0);

			// MemorySegment rollFloat = rollArray.asSlice(i * ValueLayout.JAVA_FLOAT.byteSize(), ValueLayout.JAVA_FLOAT.byteSize());
			// float roll = (float) HFFaceEulerAngle.ROLL_HANDLE.get(rollFloat, 0);
			//
			// MemorySegment yawFloat = yawArray.asSlice(i * ValueLayout.JAVA_FLOAT.byteSize(), ValueLayout.JAVA_FLOAT.byteSize());
			// float yaw = (float) HFFaceEulerAngle.YAW_HANDLE.get(yawFloat, 0);
			//
			// MemorySegment pitchFloat = pitchArray.asSlice(i * ValueLayout.JAVA_FLOAT.byteSize(), ValueLayout.JAVA_FLOAT.byteSize());
			// float pitch = (float) HFFaceEulerAngle.PITCH_HANDLE.get(pitchFloat, 0);
			//

			float roll = rollArray.getAtIndex(ValueLayout.JAVA_FLOAT, i);
			float yaw = yawArray.getAtIndex(ValueLayout.JAVA_FLOAT, i);
			float pitch = pitchArray.getAtIndex(ValueLayout.JAVA_FLOAT, i);

			detections.add(new Detection(new BoundingBox(x, y, width, height), conf, new FaceAngles(roll, yaw, pitch)));
			// System.out.printf("Face[%d] - x: %d, y: %d, width: %d, height: %d - Conf: %.2f %n", i, x, y, width, height, conf);
		}

		return detections;
	}

	private static SymbolLookup loadLib() {

		String inspirefaceLib = System.mapLibraryName("InspireFace");
		String jinspirefaceLib = System.mapLibraryName("jinspireface");

		extractAndLoad(inspirefaceLib);
		extractAndLoad(jinspirefaceLib);
		// inspirefaceLib = SymbolLookup.libraryLookup(libPath, arena);
		return SymbolLookup.loaderLookup();
	}

	private static void extractAndLoad(String name) {
		String os = System.getProperty("os.name", "generic")
			.toLowerCase(Locale.ENGLISH);
		String libpath = "";
		if (os.contains("linux")) {
			libpath = "/native" + File.separator + "linux" + File.separator + name;
		} else if (os.contains("mac")) {
			libpath = "/native" + File.separator + "macosx" + File.separator + name;
		} else {
			throw new java.lang.UnsupportedOperationException(
				os + " is not supported. Try to recompile jinspireface on your machine and then use it.");
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
		} catch (Exception e) {
			logger.error("Failed to load native lib: " + name, e);
			throw new Inspireface4jException("Failed to load native library: " + name, e);
		}

	}

	public static InspirefaceSession session(String modelPath, int detectPixelLevel, SessionFeature... featureOptions) {
		if (inspirefaceLibrary == null) {
			inspirefaceLibrary = loadLib();
		}

		// TODO make this configureable
		int maxDetectNum = 20;
		Path mPath = Paths.get(modelPath);
		if (!Files.exists(mPath)) {
			throw new Inspireface4jException("Unable to locate model with path " + mPath);
		}

		MethodHandle createHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("createSession"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));

		try (Arena arena = Arena.ofConfined()) {
			// MemorySegment labelsPathMem = arena.allocateFrom(labelsPath);
			MemorySegment modelPathMem = arena.allocateFrom(modelPath);
			int options = toHOption(featureOptions);
			MemorySegment sessionPtr = (MemorySegment) createHandler.invoke(modelPathMem, detectPixelLevel, options, maxDetectNum);
			initialized = true;
			return new InspirefaceSession(sessionPtr);
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
			MemorySegment imageSeg = imageMat.nativePtr();
			MemorySegment multipleFaceData = (MemorySegment) detectHandler.invoke(session.data(), imageSeg, false);
			multipleFaceData = multipleFaceData.reinterpret(HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteSize());
			FaceDetections detections = mapFaceDetections(multipleFaceData);
			if (drawBoundingBoxes) {
				for (Detection detection : detections) {
					BoundingBox box = detection.box();
					FaceAngles angles = detection.angles();

					float delta = angles.delta(30.0f);
					int r = (int) (255 * delta);
					int g = (int) (255 * (1-delta));
					int b = (int) (0);

					Scalar color = new Scalar(b, g, r, 0);
					CVUtils.drawRect(imageMat, box.x, box.y, box.width, box.height, color);
				}
			}
			return detections;
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to invoke detection", t);
		}

	}

	public static List<FaceLandmark> landmarks(InspirefaceSession session, Mat imageMat, FaceDetections detections, int faceNr,
		boolean drawLandmarks) {
		checkInitialized();

		MethodHandle landmarksHandler = linker
			.downcallHandle(
				inspirefaceLibrary.findOrThrow("faceLandmarks"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
					ValueLayout.JAVA_BOOLEAN));

		try {
			MemorySegment imageSeg = imageMat.nativePtr();
			MemorySegment pointData = (MemorySegment) landmarksHandler.invoke(session.data(), detections.data().segment(), imageSeg, faceNr,
				false);
			List<FaceLandmark> landmarks = mapPointData(pointData);

			if (drawLandmarks) {
				int b = (int) (Math.random() * 255);
				int g = (int) (Math.random() * 255);
				int r = (int) (Math.random() * 255);
				for (FaceLandmark landmark : landmarks) {
					int x = (int) landmark.getX();
					int y = (int) landmark.getY();
					// Scalar color = new Scalar(255 - colorOffset, 255 - colorOffset, 255 - colorOffset, 0);
					Scalar color = new Scalar(b, g, r, 0);
					CVUtils.drawCircle(imageMat, x, y, 1, color);
				}
			}
			return landmarks;
		} catch (Throwable t) {
			throw new Inspireface4jException("Failed to invoke detection", t);
		}

	}

	private static List<FaceLandmark> mapPointData(MemorySegment pointData) {
		HFLandmarkData data = new HFLandmarkData(pointData);
		return data.points();
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
			MemorySegment imageAttr = imageMat.nativePtr();
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
			MemorySegment imageAttr = imageMat.nativePtr();
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

	private static int toHOption(SessionFeature[] featureOptions) {
		int f = 0;
		for (SessionFeature option : featureOptions) {
			f |= option.getValue();
		}
		return f;
	}

}
