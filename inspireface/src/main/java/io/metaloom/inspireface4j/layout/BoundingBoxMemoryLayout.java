package io.metaloom.inspireface4j.layout;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public class BoundingBoxMemoryLayout {

	public static final GroupLayout BOUNDING_BOX_LAYOUT = MemoryLayout.structLayout(
		JAVA_INT.withName("x"),
		JAVA_INT.withName("y"),
		JAVA_INT.withName("width"),
		JAVA_INT.withName("height"));

	// VarHandles for BoundingBox
	public static final VarHandle X_HANDLE = BOUNDING_BOX_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("x"));
	public static final VarHandle Y_HANDLE = BOUNDING_BOX_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("y"));
	public static final VarHandle WIDTH_HANDLE = BOUNDING_BOX_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("width"));
	public static final VarHandle HEIGHT_HANDLE = BOUNDING_BOX_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("height"));

	// Set BoundingBox fields
	public static void setBoundingBox(MemorySegment segment, int x, int y, int width, int height) {
		X_HANDLE.set(segment, x);
		Y_HANDLE.set(segment, y);
		WIDTH_HANDLE.set(segment, width);
		HEIGHT_HANDLE.set(segment, height);
	}

	public static int getX(MemorySegment segment) {
		return (int) X_HANDLE.get(segment, 0);
	}

	public static int getY(MemorySegment segment) {
		return (int) Y_HANDLE.get(segment, 0);
	}

	public static int getWidth(MemorySegment segment) {
		return (int) WIDTH_HANDLE.get(segment, 0);
	}

	public static int getHeight(MemorySegment segment) {
		return (int) HEIGHT_HANDLE.get(segment);
	}

	public static long size() {
		return BOUNDING_BOX_LAYOUT.byteSize();
	}
}
