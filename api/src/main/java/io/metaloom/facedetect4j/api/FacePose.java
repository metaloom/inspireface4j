package io.metaloom.facedetect4j.api;

/**
 * Head orientation in degrees, as three Euler angles.
 *
 * <pre>
 *   roll   in-plane tilt.        positive = head tilted toward the image's right shoulder
 *   yaw    turn, left/right.     positive = face turned toward the image's right
 *   pitch  nod, up/down.         positive = looking up
 * </pre>
 *
 * <h2>The three are not equally trustworthy, and not equally important</h2>
 * How a pose is obtained decides what it is worth. A detector with a dedicated pose head (as
 * InspireFace has) reports measured angles. Derived from five keypoints
 * ({@link Landmarks#estimatePose()}) only <b>roll is exact</b> — it is the angle of the line
 * between the eyes and needs no model of the face at all. Yaw and pitch are inferred from where the
 * nose tip sits relative to the eye-mouth axis, which requires assuming how far an average nose
 * protrudes: the ordering is dependable, the absolute degrees are not.
 *
 * <p>
 * The distinction that matters downstream is not accuracy but <i>correctability</i>. Roll is
 * in-plane, so alignment rotates it away for free and a rolled face embeds exactly as well as an
 * upright one. Yaw and pitch are out-of-plane: no 2D warp recovers the half of the face that has
 * rotated out of view, and the embedding degrades no matter how good the alignment is. That is why
 * {@link #isFrontal(double)} ignores roll.
 */
public record FacePose(double roll, double yaw, double pitch) {

	/**
	 * Whether an embedding taken from this face is worth trusting.
	 *
	 * <p>
	 * Deliberately ignores roll — see the class note. Cosine similarity against a frontal reference
	 * falls off steeply with yaw well before a detector stops finding the face at all: on the
	 * rotation clip in the yunet examples the detector still scores 0.8 on frames whose embedding
	 * has gone orthogonal to the same person's frontal one. Gating on pose is the fix; gating on
	 * detection score is not.
	 *
	 * @param maxDegrees the out-of-plane budget, applied to yaw and pitch separately. 30 is a
	 *                   reasonable starting point for a recognition pipeline.
	 */
	public boolean isFrontal(double maxDegrees) {
		return Math.abs(yaw) <= maxDegrees && Math.abs(pitch) <= maxDegrees;
	}

	/** The larger of |yaw| and |pitch|, for ranking frames by how usable they are. */
	public double outOfPlaneDeviation() {
		return Math.max(Math.abs(yaw), Math.abs(pitch));
	}

	@Override
	public String toString() {
		return String.format("roll %+.1f, yaw %+.1f, pitch %+.1f", roll, yaw, pitch);
	}
}
