package io.metaloom.inspireface4j.layout;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class HFMultipleFaceDataLayout {

	
	public static final StructLayout HFFACE_EULER_ANGLE_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.ADDRESS.withName("roll"),
		ValueLayout.ADDRESS.withName("yaw"),
		ValueLayout.ADDRESS.withName("pitch"));

	
	public static final GroupLayout DETECTION_ARRAY_LAYOUT2 = MemoryLayout.structLayout(
		   ValueLayout.JAVA_INT.withName("detectedNum")   // HInt32
	);

	
	public static final GroupLayout DETECTION_ARRAY_LAYOUT = MemoryLayout.structLayout(
		   ValueLayout.JAVA_INT.withName("detectedNum"),   // HInt32
//		   JAVA_INT.withName("padding1"),
		   MemoryLayout.paddingLayout(4),
		    ValueLayout.ADDRESS.withName("rects"),         // HFaceRect*
		    ValueLayout.ADDRESS.withName("trackIds"),      // HInt32*
		    ValueLayout.ADDRESS.withName("detConfidence"), // HFloat*
		    HFFACE_EULER_ANGLE_LAYOUT.withName("angles"),  // embedded struct
		    ValueLayout.ADDRESS.withName("tokens")         // HFFaceBasicToken*
	);

	public static final GroupLayout FACE_RECT_LAYOUT = MemoryLayout.structLayout(
		JAVA_INT.withName("x"),
		JAVA_INT.withName("y"),
		JAVA_INT.withName("width"),
		JAVA_INT.withName("height"));

	static final StructLayout HFFACE_BASIC_TOKEN_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.JAVA_INT.withName("size"),
		 JAVA_INT.withName("padding1"),
		ValueLayout.ADDRESS.withName("data"));

//
//	// VarHandles
//	public static final VarHandle DETECTED_NUMS_HANDLER = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("detectedNum"));
//	public static final VarHandle RECTS_HANDLER = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("rects"));
//
//	// Set BoundingBox fields
//	public static void setBoundingBox(MemorySegment segment, int x, int y, int width, int height) {
//		DETECTED_NUMS_HANDLER.set(segment, x);
//		RECTS_HANDLER.set(segment, y);
//	}
//
//	public static int getDetectedNums(MemorySegment segment) {
//		return (int) DETECTED_NUMS_HANDLER.get(segment, 0);
//	}
//
//	public static int getFaceRects(MemorySegment segment) {
//		return (int) RECTS_HANDLER.get(segment, 0);
//	}
//
	public static long size() {
		return DETECTION_ARRAY_LAYOUT.byteSize();
	}
}
