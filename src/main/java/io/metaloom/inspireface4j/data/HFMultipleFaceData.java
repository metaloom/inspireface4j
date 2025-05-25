package io.metaloom.inspireface4j.data;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class HFMultipleFaceData {

	public static final GroupLayout DETECTION_ARRAY_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.JAVA_INT.withName("detectedNum"), // HInt32
		// JAVA_INT.withName("padding1"),
		MemoryLayout.paddingLayout(4),
		ValueLayout.ADDRESS.withName("rects"), // HFaceRect*
		ValueLayout.ADDRESS.withName("trackIds"), // HInt32*
		ValueLayout.ADDRESS.withName("detConfidence"), // HFloat*
		HFFaceEulerAngle.HFFACE_EULER_ANGLE_LAYOUT.withName("angles"), // embedded struct
		ValueLayout.ADDRESS.withName("tokens") // HFFaceBasicToken*
	);

	public static final VarHandle DETECTED_NUM_HANDLE = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("detectedNum"));

	public static final VarHandle RECTS_HANDLER = DETECTION_ARRAY_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("rects"));

	private MemorySegment segment;

	public HFMultipleFaceData(MemorySegment segment) {
		this.segment = segment.reinterpret(DETECTION_ARRAY_LAYOUT.byteSize());
	}

	public int detectedNum() {
		return (int) DETECTED_NUM_HANDLE.get(segment, 0);
	}

	public MemorySegment rectsMemory() {

		MemorySegment rectsPointer = segment.get(ValueLayout.ADDRESS,
			HFMultipleFaceData.DETECTION_ARRAY_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement("rects")));
		// you need to calculate this
		// // offset
		MemorySegment rectsArray = MemorySegment.ofAddress(rectsPointer.address());
		return rectsArray;
		// return (MemorySegment) RECTS_HANDLER.get(segment, 0);
	}

	public static long size() {
		return DETECTION_ARRAY_LAYOUT.byteSize();
	}

	public MemorySegment segment() {
		return segment;
	}

}
