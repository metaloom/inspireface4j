package io.metaloom.inspireface4j.layout;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public class DetectionMemoryLayout {

	// Memory Layout for Detection
	public static final GroupLayout DETECTION_LAYOUT = MemoryLayout.structLayout(
		BoundingBoxMemoryLayout.BOUNDING_BOX_LAYOUT.withName("box"),
		JAVA_FLOAT.withName("conf"),
		JAVA_INT.withName("classId"));

	// VarHandles for Detection
	public static final VarHandle CONF_HANDLE = DETECTION_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("conf"));
	public static final VarHandle CLASS_ID_HANDLE = DETECTION_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("classId"));

	// Get Detection confidence
	public static float getConf(MemorySegment segment) {
		return (float) CONF_HANDLE.get(segment, 0);
	}

	public static int getClassId(MemorySegment segment) {
		return (int) CLASS_ID_HANDLE.get(segment, 0);
	}

	public static long size() {
		return DETECTION_LAYOUT.byteSize();
	}

}
