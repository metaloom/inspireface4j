package io.metaloom.inspireface4j.data;

import java.util.ArrayList;

import io.metaloom.inspireface4j.Detection;
import io.metaloom.inspireface4j.data.internal.HFMultipleFaceData;

public class FaceDetections extends ArrayList<Detection> {

	private static final long serialVersionUID = -4970826527655238655L;
	private HFMultipleFaceData data;

	public FaceDetections(HFMultipleFaceData data) {
		this.data = data;
	}
	
	public HFMultipleFaceData data() {
		return data;
	}

}
