package io.metaloom.inspireface4j;

public enum SessionFeature {

	ENABLE_NONE(0x00000000),

	///< Flag to enable face recognition feature.
	ENABLE_FACE_RECOGNITION(0x00000002),

	///< Flag to enable RGB liveness detection feature.
	ENABLE_LIVENESS(0x00000004),

	///< Flag to enable IR (Infrared) liveness detection feature.
	ENABLE_IR_LIVENESS(0x00000008),

	///< Flag to enable mask detection feature.
	ENABLE_MASK_DETECT(0x00000010),

	///< Flag to enable face attribute prediction feature.
	ENABLE_FACE_ATTRIBUTE(0x00000020),

	///< Flag to enable face quality assessment feature.
	ENABLE_QUALITY(0x00000080),

	///< Flag to enable interaction feature.
	ENABLE_INTERACTION(0x00000100),

	///< Flag to enable face pose estimation feature.
	ENABLE_FACE_POSE(0x00000200);

	int value;

	private SessionFeature(int value) {
		this.value = value;
	}

	public int getValue() {
		return value;
	}
}
