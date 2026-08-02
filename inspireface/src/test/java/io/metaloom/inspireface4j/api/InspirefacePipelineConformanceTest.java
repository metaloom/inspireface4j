package io.metaloom.inspireface4j.api;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import io.metaloom.facedetect4j.api.AbstractFacePipelineTest;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.video4j.Video4j;

/**
 * The shared {@link FacePipeline} contract, run against InspireFace.
 *
 * <p>
 * The interesting rows are the negative ones: this backend must <em>refuse</em> to embed a supplied
 * crop rather than return something plausible from an alignment it did not perform.
 */
class InspirefacePipelineConformanceTest extends AbstractFacePipelineTest {

	private static final Path PACK = Path.of("packs/Pikachu");

	@Override
	protected void assumeBackendAvailable() {
		assumeTrue(Files.isReadable(PACK), "model pack missing: " + PACK + " -- see the README");
		// Video4j owns the OpenCV native load that MatProvider depends on.
		Video4j.init();
	}

	@Override
	protected FacePipeline pipeline() {
		return InspirefacePipeline.open(PACK.toString(), 640);
	}
}
