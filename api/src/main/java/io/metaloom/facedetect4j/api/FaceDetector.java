package io.metaloom.facedetect4j.api;

import java.util.List;

/**
 * Finds faces in an image.
 *
 * <p>
 * Implementations return <b>every</b> face above the configured score threshold, unsorted and
 * uncapped. Selecting a primary face, or limiting to N, is the caller's policy and differs by use
 * case — a portrait pipeline wants the largest central face, a crowd counter wants all of them.
 * Baking that choice into the detector removes information the caller cannot recover.
 */
public interface FaceDetector extends AutoCloseable {

	List<Face> detect(FaceImage image);

	/** Device this detector actually resolved to. Never a surprise CPU fallback. */
	Device device();

	@Override
	void close();
}
