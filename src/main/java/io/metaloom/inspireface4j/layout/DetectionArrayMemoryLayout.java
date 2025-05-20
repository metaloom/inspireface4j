package io.metaloom.inspireface4j.layout;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class DetectionArrayMemoryLayout {

	public static final GroupLayout DETECTION_ARRAY_LAYOUT = MemoryLayout.structLayout(
		JAVA_INT.withName("detectedNum"),
		ValueLayout.ADDRESS.withName("rects")
//	    HFaceRect *rects;          ///< Array of bounding rectangles for each face.
//	    HInt32 *trackIds;          ///< Array of track IDs for each face.
//	    HFloat *detConfidence;     ///< Array of detection confidence for each face.
//	    HFFaceEulerAngle angles;   ///< Euler angles for each face.
//	    PHFFaceBasicToken tokens;  ///< Tokens associated with each face.
		);
	
//
//typedef struct HFaceRect {
//    HInt32 x;             ///< X-coordinate of the top-left corner of the rectangle.
//    HInt32 y;             ///< Y-coordinate of the top-left corner of the rectangle.
//    HInt32 width;         ///< Width of the rectangle.
//    HInt32 height;        ///< Height of the rectangle.
//} HFaceRect;         ///< Rectangle representing a face region.

	// VarHandles for BoundingBox
	public static final VarHandle COUNT_HANDLER = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("count"));
	public static final VarHandle DATA_HANDLER = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("data"));

	// Set BoundingBox fields
	public static void setBoundingBox(MemorySegment segment, int x, int y, int width, int height) {
		COUNT_HANDLER.set(segment, x);
		DATA_HANDLER.set(segment, y);
	}

	public static int getCount(MemorySegment segment) {
		return (int) COUNT_HANDLER.get(segment, 0);
	}

	public static int getData(MemorySegment segment) {
		return (int) DATA_HANDLER.get(segment, 0);
	}

	public static long size()   {
		return DETECTION_ARRAY_LAYOUT.byteSize();
	}
}
