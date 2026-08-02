package io.metaloom.inspireface4j.data.internal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.*;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public class HFFaceFeature {

	public static final GroupLayout FACE_FEATURE_LAYOUT = MemoryLayout.structLayout(
		JAVA_INT.withName("size"), // HInt32
		MemoryLayout.paddingLayout(4),
		ADDRESS.withName("dataPtr") // HPFloat
	);

	public static final VarHandle SIZE_HANDLE = FACE_FEATURE_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("size"));
	public static final VarHandle DATA_PTR_HANDLE = FACE_FEATURE_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("dataPtr"));

	private MemorySegment segment;

	public HFFaceFeature(MemorySegment segment) {
		this.segment = segment.reinterpret(FACE_FEATURE_LAYOUT.byteSize());
	}

	public int size() {
		return (int) SIZE_HANDLE.get(segment, 0);
	}

	public float[] data() {
		int size = size();
		float[] data = new float[size];
		for (int i = 0; i < size; i++) {
			MemorySegment dataPtr = (MemorySegment) DATA_PTR_HANDLE.get(segment, 0);
			dataPtr = dataPtr.reinterpret(size * JAVA_FLOAT.byteSize());
			float component = dataPtr.get(JAVA_FLOAT, (long) i * JAVA_FLOAT.byteSize());
			data[i] = component;
		}
		return data;
	}

	public MemorySegment segment() {
		return segment;
	}

}
