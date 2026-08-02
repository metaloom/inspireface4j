package io.metaloom.inspireface4j.data.internal;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class HFFaceEulerAngle {

	public static final StructLayout HFFACE_EULER_ANGLE_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.ADDRESS.withName("roll"),
		ValueLayout.ADDRESS.withName("yaw"),
		ValueLayout.ADDRESS.withName("pitch"));

	public static final VarHandle ROLL_HANDLE = HFFACE_EULER_ANGLE_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("roll"));
	public static final VarHandle YAW_HANDLE = HFFACE_EULER_ANGLE_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("yaw"));
	public static final VarHandle PITCH_HANDLE = HFFACE_EULER_ANGLE_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("pitch"));

	private MemorySegment segment;

	public HFFaceEulerAngle(MemorySegment segment) {
		this.segment = segment;
	}

	public float roll() {
		return (float) ROLL_HANDLE.get(segment, 0);
	}

	public float yaw() {
		return (float) YAW_HANDLE.get(segment, 0);
	}

	public float pitch() {
		return (float) PITCH_HANDLE.get(segment, 0);
	}

	@Override
	public String toString() {
		return String.format("[EulerAngel: %.2f,%.2f,%.2f", roll(), yaw(), pitch());
	}
}
