package io.metaloom.inspireface4j.data.internal;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;

import io.metaloom.inspireface4j.FaceLandmark;

public class HFLandmarkData {

	public static final GroupLayout FACE_LANDMARK_DATA_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.JAVA_INT.withName("size"), // HInt32
		MemoryLayout.paddingLayout(4),
		ValueLayout.ADDRESS.withName("points") // HPoint2f*
	);

	public static final VarHandle SIZE_HANDLE = FACE_LANDMARK_DATA_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("size"));
	public static final VarHandle POINTS_HANDLE = FACE_LANDMARK_DATA_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("points"));

	private MemorySegment segment;

	public HFLandmarkData(MemorySegment segment) {
		this.segment = segment.reinterpret(FACE_LANDMARK_DATA_LAYOUT.byteSize());
	}

	public int size() {
		return (int) SIZE_HANDLE.get(segment, 0);
	}

	public List<FaceLandmark> points() {
		int size = size();

		List<FaceLandmark> landmarks = new ArrayList<>();

		// Resolve the pointer to the points data
		MemorySegment pointsPtr = (MemorySegment) POINTS_HANDLE.get(segment, 0);
		pointsPtr = pointsPtr.reinterpret(size * 2 * JAVA_FLOAT.byteSize());

		for (int i = 0; i < size; i++) {
			float pointValueX = pointsPtr.get(JAVA_FLOAT, (long) i * JAVA_FLOAT.byteSize());
			float pointValueY = pointsPtr.get(JAVA_FLOAT, (long) (i + 1) * JAVA_FLOAT.byteSize());
			landmarks.add(new FaceLandmark(pointValueX, pointValueY));
		}
		return landmarks;
	}

}
