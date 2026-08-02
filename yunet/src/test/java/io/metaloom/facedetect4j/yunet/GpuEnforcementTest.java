package io.metaloom.facedetect4j.yunet;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.facedetect4j.yunet.onnx.OrtRuntime;

/**
 * The "GPU is enforced, never silently substituted" contract.
 *
 * <p>
 * Deliberately <b>not</b> in {@link Yunet4jTest}: that class now skips when CUDA is unavailable,
 * which is exactly the condition the refusal path needs, so the assertion could never have run
 * there. Kept separate so each half of the contract executes on the machine that can prove it.
 */
class GpuEnforcementTest {

	private static final Path MODELS = Path.of("models");

	@BeforeAll
	static void requireModels() {
		assumeTrue(Files.isReadable(MODELS.resolve(Yunet4j.SFACE)),
			"models/ not populated -- see Yunet4j.downloadHint()");
	}

	@Test
	@DisplayName("without CUDA: refuses rather than falling back, and says how to fix it")
	void refusesWhenCudaIsMissing() {
		assumeTrue(!OrtRuntime.cudaAvailable(),
			"CUDA is available here, so the refusal path cannot be exercised");

		// A silent CPU fallback turns a deployment error into a performance mystery: nothing
		// fails, nothing logs, and the only symptom is throughput an order of magnitude low.
		assertThatThrownBy(() -> Yunet4j.pipeline(MODELS, Device.cuda()))
			.isInstanceOf(FaceException.class)
			.hasMessageContaining("will not fall back to CPU silently")
			// The message has to carry the remedy; this is the one place a user meets it.
			.hasMessageContaining("setup-cuda.sh");
	}

	@Test
	@DisplayName("with CUDA: the pipeline reports the GPU it actually attached to")
	void reportsTheGpuItGot() {
		assumeTrue(OrtRuntime.cudaAvailable(), "no CUDA provider");
		try (FacePipeline p = Yunet4j.pipeline(MODELS)) {
			assertThat(p.device().isCuda()).isTrue();
			assertThat(p.device().ordinal()).isEqualTo(0);
		}
	}

	@Test
	@DisplayName("CPU is reachable, but only by asking for it")
	void cpuIsAnExplicitOptOut() {
		// Runs on any machine: the escape hatch has to work whether or not a GPU is present,
		// otherwise the "pass Device.cpu()" advice in the error message is untested.
		try (FacePipeline p = Yunet4j.pipeline(MODELS, Device.cpu())) {
			assertThat(p.device().isCuda()).isFalse();
			assertThat(p.device().toString()).isEqualTo("cpu");
			assertThat(p.dimensions()).isEqualTo(128);
		}
	}
}
