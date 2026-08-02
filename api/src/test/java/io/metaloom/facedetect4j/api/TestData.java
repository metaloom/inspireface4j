package io.metaloom.facedetect4j.api;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Locator for the shared media under {@code facedetect4j/testdata}.
 *
 * <p>
 * The files live once at the reactor root rather than once per module: they are large (the two
 * videos alone are 43 MB), several modules want the same faces, and comparing backends is only
 * meaningful on identical input. Keeping per-module copies guarantees they drift.
 *
 * <p>
 * Shipped in this module's <b>test-jar</b>, so the backends depend on it with
 * {@code <type>test-jar</type><scope>test</scope>} and it never reaches a consumer's classpath.
 * The media itself is not packaged — only this locator is. That is deliberate: 45 MB of video does
 * not belong in a jar, and tests read it straight off disk.
 */
public final class TestData {

	private static final Path ROOT = locateRoot();

	private TestData() {
	}

	/**
	 * Walk up from the working directory looking for {@code testdata}.
	 *
	 * <p>
	 * Not a fixed {@code ../testdata}: Maven runs tests with the working directory set to the
	 * module, but IDEs commonly use the reactor root or the project root instead. A hardcoded
	 * relative path works under {@code mvn} and then fails in the IDE for reasons that look like
	 * missing files rather than a wrong CWD.
	 */
	private static Path locateRoot() {
		Path dir = Path.of("").toAbsolutePath();
		for (int i = 0; dir != null && i < 6; i++, dir = dir.getParent()) {
			Path candidate = dir.resolve("testdata");
			if (Files.isDirectory(candidate)) {
				return candidate;
			}
		}
		return null;
	}

	/** True when the media is reachable. Tests should skip, not fail, when it is not. */
	public static boolean available() {
		return ROOT != null;
	}

	public static Path root() {
		if (ROOT == null) {
			throw new IllegalStateException(
				"testdata/ not found above " + Path.of("").toAbsolutePath()
					+ " -- run tests from within the facedetect4j checkout");
		}
		return ROOT;
	}

	/** A still under {@code testdata/images}. */
	public static Path image(String name) {
		return root().resolve("images").resolve(name);
	}

	/** A clip under {@code testdata/video}. */
	public static Path video(String name) {
		return root().resolve("video").resolve(name);
	}

	// Faces used across modules. Named by what they are for, since the filenames are provenance
	// (photographer credits in testdata/SOURCES.txt) and say nothing about content.

	/** Frontal portrait, neutral expression. The default single-face fixture. */
	public static final String IMG_FACE_NEUTRAL = "face_neutral.jpg";
	/** Same identity as {@link #IMG_FACE_NEUTRAL}, smiling — the within-identity pair. */
	public static final String IMG_FACE_HAPPY = "face_happy.jpg";
	/** A different identity, for the between-identity comparison. */
	public static final String IMG_FACE_OTHER_IDENTITY = "jeri-ryan-1.jpg";
	/** 24 faces, ground truth for multi-face detection. */
	public static final String IMG_COLLAGE_24_FACES = "collage-24-faces.jpeg";

	public static final String IMG_FACE_RASTER_HQ = "pexels-olly-3812743.jpg";
	public static final String IMG_FACE_RASTER_1K = "pexels-olly-3812743_1k.jpg";
	public static final String IMG_FACE_RASTER_1K_LOWER = "pexels-olly-3812743_1k_lower.jpg";
	public static final String IMG_FACE_RASTER_1K_UPPER = "pexels-olly-3812743_1k_upper.jpg";
	public static final String IMG_FACE_OCCLUDED = "pexels-olly-3812743_occluded.jpg";
	public static final String IMG_FACE_OCCLUDED_HQ = "pexels-olly-3812743_occluded_hq.jpg";
	public static final String IMG_FACE_NOT_OCCLUDED = "pexels-olly-3812743_not_occluded.jpg";

	public static final String IMG_ASIAN_WOMAN = "pexels-carol-wd-1531174-3284696.jpg";
	public static final String IMG_MEXICAN_MEN = "pexels-cristian-rojas-7195925.jpg";
	public static final String IMG_BLACK_MEN = "pexels-kindelmedia-8173262.jpg";
	public static final String IMG_CHILD = "pexels-mikhail-nilov-6957906.jpg";

	public static final String VID_FACE_ROTATE_1 = "8090198-hd_1366_720_25fps.mp4";
	public static final String VID_2 = "8090198-uhd_4096_2160_25fps.mp4";
}
