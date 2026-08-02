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

	/**
	 * Check whether any of the angles (roll, yaw, pitch) exceeds the given angle.
	 * 
	 * @param limit
	 * @return
	 */
	public boolean exceeds(float limit) {
		if (Math.abs(roll) >= limit) {
			return true;
		}
		if (Math.abs(yaw) >= limit) {
			return true;
		}
		if (Math.abs(pitch) >= limit) {
			return true;
		}
		return false;
	}

	/**
	 * Compute the delta factor of roll, yaw and pitch to the given limit.
	 * 
	 * @param limit
	 * @return
	 */
	public float delta(float limit) {
		float delta = 0;
		float d = Math.abs(roll) / limit;
		if (d > delta) {
			delta = d;
		}
		d = Math.abs(yaw) / limit;
		if (d > delta) {
			delta = d;
		}
		d = Math.abs(pitch) / limit;
		if (d > delta) {
			delta = d;
		}

		if (delta >= 1) {
			delta = 1;
		}

		return delta;
	}

	@Override
	public String toString() {
		return "[roll: %.2f, pitch: %.2f, yaw: %.2f]".formatted(roll, pitch, yaw);
	}
}
