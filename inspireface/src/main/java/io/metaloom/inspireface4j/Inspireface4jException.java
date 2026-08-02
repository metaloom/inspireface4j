package io.metaloom.inspireface4j;

public class Inspireface4jException extends RuntimeException {

	private static final long serialVersionUID = -6761918386931878598L;

	public Inspireface4jException(String msg, Throwable t) {
		super(msg, t);
	}

	public Inspireface4jException(String msg) {
		super(msg);
	}

}
