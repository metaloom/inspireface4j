package io.metaloom.facedetect4j.yunet;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import io.metaloom.facedetect4j.api.AbstractFacePipelineTest;
import io.metaloom.facedetect4j.api.FacePipeline;

/** The shared {@link FacePipeline} contract, run against YuNet + SFace on the GPU. */
class YunetPipelineConformanceTest extends AbstractFacePipelineTest {

	private static final Path MODELS = Path.of("models");

	@Override
	protected void assumeBackendAvailable() {
		assumeTrue(Files.isReadable(MODELS.resolve(Yunet4j.SFACE)),
			"models/ not populated -- see Yunet4j.downloadHint()");
		assumeTrue(io.metaloom.facedetect4j.yunet.onnx.OrtRuntime.cudaAvailable(),
			"no CUDA provider -- this backend does not fall back silently");
	}

	@Override
	protected FacePipeline pipeline() {
		return Yunet4j.pipeline(MODELS);
	}
}
