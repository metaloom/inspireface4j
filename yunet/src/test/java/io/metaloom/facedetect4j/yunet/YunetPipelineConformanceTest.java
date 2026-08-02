package io.metaloom.facedetect4j.yunet;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import io.metaloom.facedetect4j.api.AbstractFacePipelineTest;
import io.metaloom.facedetect4j.api.FacePipeline;

/** The shared {@link FacePipeline} contract, run against YuNet + SFace on the GPU. */
class YunetPipelineConformanceTest extends AbstractFacePipelineTest {

	private static final Path MODELS = Path.of("models");

	@Override
	protected void assumeBackendAvailable() {
		assumeTrue(GpuProbe.usable(MODELS), GpuProbe.skipReason());
	}

	@Override
	protected FacePipeline pipeline() {
		return Yunet4j.pipeline(MODELS);
	}
}
