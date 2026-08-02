package io.metaloom.inspireface4j.data.internal;

import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class PHFFaceBasicToken {

	public static final StructLayout HFFACE_BASIC_TOKEN_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.JAVA_INT.withName("size"),
		JAVA_INT.withName("padding1"),
		ValueLayout.ADDRESS.withName("data"));

	private MemorySegment segment;

	public static final VarHandle SIZE_HANDLE = HFFACE_BASIC_TOKEN_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("size"));

	public PHFFaceBasicToken(MemorySegment segment) {
		this.segment = segment;
	}

	public int size() {
		return (int) SIZE_HANDLE.get(segment, 0);
	}

	public void data() {

	}
}
