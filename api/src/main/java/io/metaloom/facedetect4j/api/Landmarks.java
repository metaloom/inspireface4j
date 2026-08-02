package io.metaloom.facedetect4j.api;

/**
 * The five facial keypoints, in <b>ArcFace order</b> and absolute source-image pixels:
 *
 * <pre>
 *   0  left eye           (image-left, i.e. the subject's right)
 *   1  right eye
 *   2  nose tip
 *   3  left mouth corner
 *   4  right mouth corner
 * </pre>
 *
 * This order is the interop contract for alignment, not a naming convention. Permuting it produces
 * a crop that still looks like a face and still embeds without error — it just scores worse, with
 * nothing in the pipeline reporting a problem.
 */
public record Landmarks(float[] xy) {

	public Landmarks {
		if (xy.length != 10) {
			throw new IllegalArgumentException("expected 5 points (10 floats), got " + xy.length);
		}
	}

	public float x(int i) {
		return xy[2 * i];
	}

	public float y(int i) {
		return xy[2 * i + 1];
	}

	public Landmarks scaled(float f) {
		float[] o = new float[10];
		for (int i = 0; i < 10; i++) {
			o[i] = xy[i] * f;
		}
		return new Landmarks(o);
	}

	/**
	 * Where the nose tip sits between the eye line and the mouth line on a frontal face, as a
	 * fraction. Read straight off the insightface {@code arcface_dst} template that
	 * {@link io.metaloom.facedetect4j.api.align.ArcFaceAlign} aligns to, where the nose lands on
	 * the eye-to-mouth axis to within a thousandth of a pixel — so the same template that defines
	 * "aligned" also defines "frontal", and the two cannot drift apart.
	 */
	private static final double NOSE_FRACTION = 0.49496;

	/**
	 * Interocular distance divided by how far the nose tip protrudes from the plane of the eyes and
	 * mouth. This is the one quantity a 2D keypoint set cannot supply and the whole yaw/pitch
	 * estimate turns on it: the nose is the only one of the five points that is not roughly
	 * coplanar with the others, which is exactly why it moves under rotation and they do not.
	 *
	 * <p>
	 * 3.0 from population anthropometry — pupillary distance about 63 mm, nose tip about 21 mm
	 * proud of the face plane. Being a population average, it sets the <i>scale</i> of the reported
	 * angles and is the reason those degrees are approximate while their ordering is not.
	 */
	private static final double NOSE_LEVER = 3.0;

	/**
	 * Interocular distance over eye-to-mouth distance on a frontal face, from the same template.
	 * Yaw foreshortens the first and leaves the second alone, so their ratio measures the turn
	 * without needing to know anything about the face's depth.
	 *
	 * <p>
	 * {@code FacePoseTest} recomputes this from {@link io.metaloom.facedetect4j.api.align.ArcFaceAlign}
	 * so it cannot drift away from the template it was read off.
	 */
	private static final double FRONTAL_EYE_MOUTH_RATIO = 0.8660777;

	/**
	 * Head orientation, estimated from the five points alone.
	 *
	 * <p>
	 * <b>Roll is exact</b>: the angle of the line between the eyes. Yaw and pitch are estimates,
	 * and the geometry behind them is worth knowing before trusting a number. Four of the five
	 * keypoints — both eyes and both mouth corners — lie close to one plane. The nose tip does not;
	 * it stands {@code interocular / NOSE_LEVER} in front of it. Rotate the head and the four
	 * coplanar points merely foreshorten, while the nose tip swings sideways by
	 * {@code protrusion * sin(angle)}. Measuring that swing against the eye-to-mouth axis therefore
	 * recovers the rotation, as {@code tan(angle) = offset / protrusion}.
	 *
	 * <p>
	 * What that costs: the answer is only as good as {@link #NOSE_LEVER} is for this particular
	 * face, so a prominent or a flat nose reads as more or less rotation than there is. Treat the
	 * ordering as sound and the degrees as within perhaps ten of the truth. Beyond roughly 60
	 * degrees of yaw the far eye is occluded and the detector is guessing at its position, so the
	 * estimate degrades exactly where it is largest — enough to know the face has turned away,
	 * which is what {@link FacePose#isFrontal(double)} needs, and not enough to drive a 3D pose.
	 *
	 * <p>
	 * For measured angles rather than estimated ones, use a detector with a pose head (InspireFace
	 * has one) or run {@code solvePnP} against a 3D face model with real camera intrinsics.
	 */
	public FacePose estimatePose() {
		double lx = x(0), ly = y(0), rx = x(1), ry = y(1);
		double dx = rx - lx, dy = ry - ly;
		double interocular = Math.hypot(dx, dy);
		double rollRad = Math.atan2(dy, dx);
		double roll = Math.toDegrees(rollRad);
		if (interocular < 1e-6) {
			// Coincident eyes: there is no face here, and no axis to measure anything against.
			return new FacePose(roll, 0, 0);
		}

		// Into the face's own frame: eye line horizontal, eye midpoint at the origin, y downward.
		// Rolling out first is what makes the two remaining angles independent of the third.
		double cos = Math.cos(rollRad), sin = Math.sin(rollRad);
		double eyeX = (lx + rx) / 2.0, eyeY = (ly + ry) / 2.0;
		double mouthDx = (x(3) + x(4)) / 2.0 - eyeX, mouthDy = (y(3) + y(4)) / 2.0 - eyeY;
		double mouthX = mouthDx * cos + mouthDy * sin;
		double mouthY = -mouthDx * sin + mouthDy * cos;
		double noseDx = x(2) - eyeX, noseDy = y(2) - eyeY;
		double noseX = noseDx * cos + noseDy * sin;
		double noseY = -noseDx * sin + noseDy * cos;

		if (mouthY < 1e-6) {
			// Mouth level with or above the eyes. Either the point order is wrong or this is not a
			// face; either way the axis below would divide by nothing.
			return new FacePose(roll, 0, 0);
		}

		// Yaw, by two measurements that fail in opposite places.
		//
		// The nose swinging off the eye-to-mouth axis is sensitive near frontal, where the eyes are
		// well separated and the axis is meaningful. It collapses at profile: once the far eye is
		// occluded the detector stacks it onto the near one, the "eye midpoint" stops being the
		// midline of the face, and the offset measured against it means nothing. Measured on the
		// rotation clip, this read a 17 degree turn on a frame that was in full profile.
		//
		// The interocular distance foreshortening against the eye-to-mouth distance is the reverse:
		// at profile it is unmistakable (the same frame's eyes had closed to 0.31 of their frontal
		// separation, i.e. a 72 degree turn) while near frontal it is noise, since acos is vertical
		// at 1 and a pixel of jitter swings it by degrees.
		//
		// So take the larger magnitude and the nose's sign. Each method under-reports where it is
		// weak and neither over-reports, so the maximum is the one that cannot be fooled into
		// calling a turned face frontal - which is the direction a gate must fail in.
		double along = noseY / mouthY;
		double axisX = mouthX * along;
		double offset = noseX - axisX;
		double byNose = Math.toDegrees(Math.atan2(NOSE_LEVER * offset, interocular));

		double eyeToMouth = Math.hypot(mouthX, mouthY);
		double byForeshortening = Math.toDegrees(Math.acos(
			Math.min(1.0, (interocular / eyeToMouth) / FRONTAL_EYE_MOUTH_RATIO)));

		double yaw = (offset < 0 ? -1 : 1) * Math.max(Math.abs(byNose), byForeshortening);

		// Pitch: the same lever, applied to the nose's displacement along that axis rather than
		// across it. Negated because a nose sitting lower than NOSE_FRACTION means looking down.
		double pitch = -Math.toDegrees(
			Math.atan2(NOSE_LEVER * (along - NOSE_FRACTION) * mouthY, interocular));

		return new FacePose(roll, yaw, pitch);
	}

	/**
	 * Cheap plausibility check on the ordering: image-left eye must sit left of the right eye, and
	 * the eyes must sit above the mouth. Useful as an assertion when integrating a new detector.
	 */
	public boolean geometryLooksSane() {
		float eyeY = (y(0) + y(1)) / 2f;
		float mouthY = (y(3) + y(4)) / 2f;
		return x(0) < x(1) && eyeY < mouthY;
	}
}
