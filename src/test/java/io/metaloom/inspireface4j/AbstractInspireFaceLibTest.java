package io.metaloom.inspireface4j;

import io.metaloom.video4j.Video4j;

public abstract class AbstractInspireFaceLibTest {

	private static String modelPath = "InspireFace/test_res/pack/Pikachu";
	//private static String modelPath = "InspireFace/test_res/pack/Megatron";
	//private static String modelPath = "InspireFace/test_res/pack/Megatron_TRT";

	static {
		Video4j.init();
		InspirefaceLib.init(modelPath, 320);
	}

}
