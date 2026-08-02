package io.metaloom.facedetect4j.api;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import io.metaloom.facedetect4j.api.align.ArcFaceAlign;

/**
 * Pose estimated from five keypoints.
 *
 * <p>
 * The fixture is the insightface {@code arcface_dst} template itself rather than hand-written
 * coordinates. That makes the frontal case a real assertion instead of a restatement of a constant:
 * if {@code NOSE_FRACTION} ever stops describing the template that alignment warps to, these fail.
 *
 * <p>
 * Roll is checked for exactness; yaw and pitch for sign, monotonicity and invariance, which is what
 * the estimate actually promises. Absolute degrees are only pinned loosely — see
 * {@link Landmarks#estimatePose()} for why they cannot be pinned tightly.
 */
class FacePoseTest {

	private static Landmarks canonical() {
		double[][] t = ArcFaceAlign.template();
		float[] xy = new float[10];
		for (int i = 0; i < 5; i++) {
			xy[2 * i] = (float) t[i][0];
			xy[2 * i + 1] = (float) t[i][1];
		}
		return new Landmarks(xy);
	}

	/** Rotate about the image origin by {@code deg}, clockwise in screen coordinates (y down). */
	private static Landmarks rotated(Landmarks lm, double deg) {
		double a = Math.toRadians(deg), c = Math.cos(a), s = Math.sin(a);
		float[] xy = new float[10];
		for (int i = 0; i < 5; i++) {
			double x = lm.x(i) - 56, y = lm.y(i) - 72;
			xy[2 * i] = (float) (x * c - y * s + 56);
			xy[2 * i + 1] = (float) (x * s + y * c + 72);
		}
		return new Landmarks(xy);
	}

	/** Move only the nose tip sideways, which is exactly what yaw does to it. */
	private static Landmarks noseShifted(Landmarks lm, double dx, double dy) {
		float[] xy = lm.xy().clone();
		xy[4] += (float) dx;
		xy[5] += (float) dy;
		return new Landmarks(xy);
	}

	@Test
	@DisplayName("the frontal eye/mouth ratio still describes the template it was read off")
	void frontalRatioMatchesTemplate() {
		Landmarks lm = canonical();
		double interocular = Math.hypot(lm.x(1) - lm.x(0), lm.y(1) - lm.y(0));
		double eyeToMouth = Math.hypot((lm.x(3) + lm.x(4)) / 2 - (lm.x(0) + lm.x(1)) / 2,
			(lm.y(3) + lm.y(4)) / 2 - (lm.y(0) + lm.y(1)) / 2);
		// The constant is private, so this asserts on the observable consequence: the template
		// itself must foreshorten by nothing, which is only true if the constant matches.
		assertThat(interocular / eyeToMouth).isCloseTo(0.8660777, within(1e-6));
	}

	@Test
	@DisplayName("eyes closing together read as a turn even when the nose says otherwise")
	void foreshorteningCatchesProfile() {
		// The profile signature measured on the rotation clip: the far eye is occluded, the
		// detector stacks it onto the near one, and the nose offset - measured against an eye
		// midpoint that is no longer the midline - reports almost nothing.
		Landmarks lm = canonical();
		float[] xy = lm.xy().clone();
		float midX = (xy[0] + xy[2]) / 2;
		xy[0] = midX - 5.5f;   // eyes closed to ~0.31 of their frontal separation
		xy[2] = midX + 5.5f;
		FacePose pose = new Landmarks(xy).estimatePose();
		assertThat(Math.abs(pose.yaw())).as("yaw from foreshortening alone").isGreaterThan(60);
		assertThat(pose.isFrontal(30)).isFalse();
	}

	@Test
	@DisplayName("the alignment template is, by definition, a frontal face")
	void canonicalIsFrontal() {
		FacePose pose = canonical().estimatePose();
		// The template is very slightly rolled: its two eye y values differ by 0.195 px.
		assertThat(pose.roll()).isCloseTo(-0.317, within(0.01));
		assertThat(pose.yaw()).as("yaw").isCloseTo(0.0, within(0.05));
		assertThat(pose.pitch()).as("pitch").isCloseTo(0.0, within(0.05));
		assertThat(pose.isFrontal(1)).isTrue();
	}

	@Nested
	@DisplayName("roll")
	class Roll {

		@Test
		@DisplayName("is exact, and recovers the rotation applied")
		void isExact() {
			for (double deg : new double[] { -75, -30, -5, 5, 30, 75 }) {
				FacePose pose = rotated(canonical(), deg).estimatePose();
				assertThat(pose.roll()).as("roll %s", deg)
					.isCloseTo(-0.317 + deg, within(0.001));
			}
		}

		@Test
		@DisplayName("does not leak into yaw or pitch: rolling out comes first")
		void doesNotLeak() {
			// A rolled frontal face must still read as frontal. Were the nose offset measured in
			// image axes rather than the face's own, a 45 degree roll alone would fake a large yaw.
			FacePose pose = rotated(canonical(), 45).estimatePose();
			assertThat(pose.yaw()).as("yaw").isCloseTo(0.0, within(0.05));
			assertThat(pose.pitch()).as("pitch").isCloseTo(0.0, within(0.05));
			assertThat(pose.isFrontal(1)).as("frontal despite 45 degrees of roll").isTrue();
		}
	}

	@Nested
	@DisplayName("yaw")
	class Yaw {

		@Test
		@DisplayName("positive means turned toward the image's right")
		void sign() {
			assertThat(noseShifted(canonical(), 6, 0).estimatePose().yaw()).isGreaterThan(20);
			assertThat(noseShifted(canonical(), -6, 0).estimatePose().yaw()).isLessThan(-20);
		}

		@Test
		@DisplayName("grows monotonically with the nose offset")
		void monotonic() {
			double previous = Double.NEGATIVE_INFINITY;
			for (double dx = -12; dx <= 12; dx += 2) {
				double yaw = noseShifted(canonical(), dx, 0).estimatePose().yaw();
				assertThat(yaw).as("dx %s", dx).isGreaterThan(previous);
				previous = yaw;
			}
		}

		@Test
		@DisplayName("is scale invariant: the same face closer to the camera is not more turned")
		void scaleInvariant() {
			Landmarks turned = noseShifted(canonical(), 6, 0);
			assertThat(turned.scaled(4f).estimatePose().yaw())
				.isCloseTo(turned.estimatePose().yaw(), within(1e-6));
		}

		@Test
		@DisplayName("a nose halfway to one eye reads as a substantial turn, not a slight one")
		void magnitudeIsPlausible() {
			// Half the interocular distance is a face in clear three-quarter view. The estimate is
			// approximate, so this pins the band rather than a value - but a number down at 15
			// degrees, or up at 89, would mean the lever is wrong.
			double half = 35.2372 / 2;
			double yaw = noseShifted(canonical(), half, 0).estimatePose().yaw();
			assertThat(yaw).isBetween(50.0, 65.0);
		}
	}

	@Nested
	@DisplayName("pitch")
	class Pitch {

		@Test
		@DisplayName("positive means looking up")
		void sign() {
			// The nose riding up toward the eye line is what looking up does to it.
			assertThat(noseShifted(canonical(), 0, -6).estimatePose().pitch()).isGreaterThan(15);
			assertThat(noseShifted(canonical(), 0, 6).estimatePose().pitch()).isLessThan(-15);
		}
	}

	@Nested
	@DisplayName("isFrontal")
	class Frontal {

		@Test
		@DisplayName("ignores roll, because alignment corrects it")
		void ignoresRoll() {
			assertThat(new FacePose(80, 0, 0).isFrontal(30)).isTrue();
			assertThat(new FacePose(0, 40, 0).isFrontal(30)).isFalse();
			assertThat(new FacePose(0, 0, -40).isFrontal(30)).isFalse();
		}

		@Test
		@DisplayName("outOfPlaneDeviation ranks by the worse of the two correctable-by-nothing axes")
		void deviation() {
			assertThat(new FacePose(90, -35, 10).outOfPlaneDeviation()).isEqualTo(35);
		}
	}

	@Nested
	@DisplayName("degenerate input")
	class Degenerate {

		@Test
		@DisplayName("coincident eyes report no rotation rather than dividing by zero")
		void coincidentEyes() {
			FacePose pose = new Landmarks(new float[] { 10, 10, 10, 10, 10, 20, 5, 30, 15, 30 })
				.estimatePose();
			assertThat(pose.yaw()).isZero();
			assertThat(pose.pitch()).isZero();
		}

		@Test
		@DisplayName("a mouth above the eyes yields no angles instead of nonsense ones")
		void invertedFace() {
			// Points in the wrong order. Better to report nothing than a confident wrong pose that
			// would pass isFrontal().
			FacePose pose = new Landmarks(new float[] { 0, 100, 40, 100, 20, 80, 5, 50, 35, 50 })
				.estimatePose();
			assertThat(pose.yaw()).isZero();
			assertThat(pose.pitch()).isZero();
		}
	}
}
