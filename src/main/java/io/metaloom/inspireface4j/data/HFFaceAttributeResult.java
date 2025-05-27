package io.metaloom.inspireface4j.data;

import java.lang.foreign.GroupLayout;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;

public class HFFaceAttributeResult {

	public static final GroupLayout FACE_ATTR_LAYOUT = MemoryLayout.structLayout(
		ValueLayout.JAVA_INT.withName("num"), // HInt32
		MemoryLayout.paddingLayout(4),
		ValueLayout.ADDRESS.withName("race"), // HPInt32*
		ValueLayout.ADDRESS.withName("gender"), // HPInt32*
		ValueLayout.ADDRESS.withName("ageBracket") // HPInt32*
	);

	public static final VarHandle NUM_HANDLE = FACE_ATTR_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("num"));
	public static final VarHandle RACE_HANDLE = FACE_ATTR_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("race"));
	public static final VarHandle GENDER_HANDLE = FACE_ATTR_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("gender"));
	public static final VarHandle AGE_BRACKET_HANDLE = FACE_ATTR_LAYOUT.varHandle(MemoryLayout.PathElement.groupElement("ageBracket"));

	private MemorySegment segment;

	public HFFaceAttributeResult(MemorySegment segment) {
		this.segment = segment.reinterpret(FACE_ATTR_LAYOUT.byteSize());
	}

	public int numFaces() {
		return (int) NUM_HANDLE.get(segment, 0);
	}

	public Race race(int faceNr) {
		MemorySegment racePtr = (MemorySegment) RACE_HANDLE.get(segment, 0);
		racePtr = racePtr.reinterpret(numFaces() * JAVA_INT.byteSize());
		int raceInt = racePtr.get(JAVA_INT, (long) faceNr * JAVA_INT.byteSize());
		// System.out.println("RaceInt: " + raceInt);

		return switch (raceInt) {
		case 0 -> Race.BLACK;
		case 1 -> Race.ASIAN;
		case 2 -> Race.LATINO_HISPANIC;
		case 3 -> Race.MIDDLE_EASTERN;
		case 4 -> Race.WHITE;
		default -> throw new IllegalArgumentException("Unexpected race number: " + raceInt);
		};
	}

	public Gender gender(int faceNr) {
		MemorySegment genderPtr = (MemorySegment) GENDER_HANDLE.get(segment, 0);
		genderPtr = genderPtr.reinterpret(numFaces() * JAVA_INT.byteSize());
		int genderInt = genderPtr.get(JAVA_INT, (long) faceNr * JAVA_INT.byteSize());
		// System.out.println("GenderInt: " + genderInt);

		return genderInt == 0 ? Gender.FEMALE : Gender.MALE;
	}

	public AgeBracket age(int faceNr) {
		MemorySegment agePtr = (MemorySegment) AGE_BRACKET_HANDLE.get(segment, 0);
		agePtr = agePtr.reinterpret(numFaces() * JAVA_INT.byteSize());
		int ageInt = agePtr.get(JAVA_INT, (long) faceNr * JAVA_INT.byteSize());
		// System.out.println("AgeInt: " + ageInt);

		return switch (ageInt) {
		case 0 -> AgeBracket.AGE_0_2;
		case 1 -> AgeBracket.AGE_3_9;
		case 2 -> AgeBracket.AGE_10_19;
		case 3 -> AgeBracket.AGE_20_29;
		case 4 -> AgeBracket.AGE_30_39;
		case 5 -> AgeBracket.AGE_40_49;
		case 6 -> AgeBracket.AGE_50_59;
		case 7 -> AgeBracket.AGE_60_69;
		case 8 -> AgeBracket.ABOVE_70;
		default -> throw new IllegalArgumentException("Unexpected age bracket number: " + ageInt);
		};
	}

}
