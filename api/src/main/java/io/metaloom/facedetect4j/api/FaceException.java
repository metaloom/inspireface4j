package io.metaloom.facedetect4j.api;

/** Anything that goes wrong loading models or running inference. */
public class FaceException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public FaceException(String message) {
		super(message);
	}

	public FaceException(String message, Throwable cause) {
		super(message, cause);
	}
}
