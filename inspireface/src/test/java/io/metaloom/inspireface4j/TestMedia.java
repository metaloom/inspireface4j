package io.metaloom.inspireface4j;

import io.metaloom.facedetect4j.api.TestData;

/**
 * Paths to the shared media in {@code facedetect4j/testdata}.
 *
 * <p>
 * Kept as a facade over {@link TestData} so the existing tests read unchanged. The media moved out
 * of this module because several backends want the same faces, and comparing backends is only
 * meaningful on identical input.
 */
public class TestMedia {

	public static final String IMG_FACE_RASTER_1K_LOWER = img(TestData.IMG_FACE_RASTER_1K_LOWER);
	public static final String IMG_FACE_RASTER_1K_RES = img(TestData.IMG_FACE_RASTER_1K);
	public static final String IMG_FACE_RASTER_1K_UPPER = img(TestData.IMG_FACE_RASTER_1K_UPPER);
	public static final String IMG_FACE_OCCLUDED = img(TestData.IMG_FACE_OCCLUDED);
	public static final String IMG_FACE_NOT_OCCLUDED = img(TestData.IMG_FACE_NOT_OCCLUDED);
	public static final String IMG_FACE_RASTER_HQ = img(TestData.IMG_FACE_RASTER_HQ);

	public static final String VID_FACE_ROTATE_1 = vid(TestData.VID_FACE_ROTATE_1);
	public static final String VID_2 = vid(TestData.VID_2);

	public static final String IMG_ASIAN_WOMAN = img(TestData.IMG_ASIAN_WOMAN);
	public static final String IMG_MEXICAN_MEN = img(TestData.IMG_MEXICAN_MEN);
	public static final String IMG_BLACK_MEN = img(TestData.IMG_BLACK_MEN);
	public static final String IMG_CHILD = img(TestData.IMG_CHILD);

	private static String img(String name) {
		return TestData.image(name).toString();
	}

	private static String vid(String name) {
		return TestData.video(name).toString();
	}
}
