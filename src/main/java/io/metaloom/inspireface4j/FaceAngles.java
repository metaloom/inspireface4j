package io.metaloom.inspireface4j;

public class FaceAngles {

	private float roll;
	private float yaw;
	private float pitch;

	public FaceAngles(float roll, float yaw, float pitch) {
		this.roll = roll;
		this.yaw = yaw;
		this.pitch = pitch;
	}

	public float roll() {
		return roll;
	}

	public float yaw() {
		return yaw;
	}

	public float pitch() {
		return pitch;
	}

}
