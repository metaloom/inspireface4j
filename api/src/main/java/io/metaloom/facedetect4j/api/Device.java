package io.metaloom.facedetect4j.api;

/**
 * Which accelerator to run on.
 *
 * <p>
 * <b>CUDA is the default and is enforced, not preferred.</b> A silent CPU fallback is the worst
 * possible behaviour for a library whose entire reason to exist is GPU throughput: everything
 * still works, nothing logs an error, and the only symptom is a frame rate an order of magnitude
 * below expectation — usually noticed in production. {@link #cpu()} exists, but you have to ask
 * for it.
 */
public record Device(Kind kind, int ordinal) {

	public enum Kind {
		CUDA, CPU
	}

	/** GPU 0. */
	public static Device cuda() {
		return new Device(Kind.CUDA, 0);
	}

	public static Device cuda(int ordinal) {
		return new Device(Kind.CUDA, ordinal);
	}

	/** Explicit opt-out of GPU. Never selected implicitly. */
	public static Device cpu() {
		return new Device(Kind.CPU, -1);
	}

	public boolean isCuda() {
		return kind == Kind.CUDA;
	}

	@Override
	public String toString() {
		return kind == Kind.CUDA ? "cuda:" + ordinal : "cpu";
	}
}
