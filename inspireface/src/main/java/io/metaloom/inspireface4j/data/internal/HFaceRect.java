package io.metaloom.inspireface4j.data.internal;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public class HFaceRect {

	public static final GroupLayout FACE_RECT_LAYOUT = MemoryLayout.structLayout(
		JAVA_INT.withName("x"),
		JAVA_INT.withName("y"),
		JAVA_INT.withName("width"),
		JAVA_INT.withName("height"));

	public static final VarHandle X_HANDLE = FACE_RECT_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("x"));
	public static final VarHandle Y_HANDLE = FACE_RECT_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("y"));
	public static final VarHandle WIDTH_HANDLE = FACE_RECT_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("width"));
	public static final VarHandle HEIGHT_HANDLE = FACE_RECT_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("height"));

	private MemorySegment segment;

	private int index;

	public HFaceRect(MemorySegment segment, int index) {
		this.index = index;
		// Compute the offset for the specific HFaceRect struct
		this.segment = segment.asSlice(index * FACE_RECT_LAYOUT.byteSize(), FACE_RECT_LAYOUT.byteSize());
	}

	@Override
	public String toString() {
		return String.format("Face[%d] - x: %d, y: %d, width: %d, height: %d%n", index(), x(), y(), width(), height());
	}

	public int index() {
		return index;
	}

	public int x() {
		return (int) X_HANDLE.get(segment, 0);
	}

	public int y() {
		return (int) Y_HANDLE.get(segment, 0);
	}

	public int width() {
		return (int) WIDTH_HANDLE.get(segment, 0);
	}

	public int height() {
		return (int) HEIGHT_HANDLE.get(segment, 0);
	}
}
