package io.metaloom.inspireface4j.layout;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class DetectionArrayMemoryLayout {

	public static final GroupLayout DETECTION_ARRAY_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.ADDRESS.withName("data"),
		JAVA_INT.withName("count"));

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
