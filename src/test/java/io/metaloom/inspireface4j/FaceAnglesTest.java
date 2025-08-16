package io.metaloom.inspireface4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class FaceAnglesTest {

	@Test
	public void testExceeds() {
		assertTrue(new FaceAngles(15f, 30f, 45f).exceeds(45f));
		assertTrue(new FaceAngles(15f, 30f, -45f).exceeds(45f));
		assertFalse(new FaceAngles(15f, 30f, -45f).exceeds(46f));
	}

	@Test
	public void testDelta() {
		assertEquals(1.0f, new FaceAngles(15f, 30f, 45f).delta(30f));
		assertEquals(0.5f, new FaceAngles(13f, 15f, 14f).delta(30f));
		assertEquals(0.23f, new FaceAngles(7f, 3f, 1f).delta(30f), 0.1f);
	}

	@Test
	public void testDeltaRGB() {
		assertRGB(0.25f, 63, 191);
		assertRGB(1.0f, 255, 0);
		assertRGB(0f, 0, 255);
	}

	private void assertRGB(float delta, int expectedRed, int expectedGreen) {
		int r = (int) (delta * 255);
		assertEquals(expectedRed, r);

		int g = (int) ((1 - delta) * 255);
		assertEquals(expectedGreen, g);
	}
}
