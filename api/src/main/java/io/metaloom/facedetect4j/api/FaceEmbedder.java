package io.metaloom.facedetect4j.api;

/**
 * Turns a face into a comparable vector.
 *
 * <p>
 * Two entry points on purpose. {@link #embed(FaceImage, Face)} is the convenient one and aligns
 * internally. {@link #embed(AlignedFace)} takes pixels the caller already aligned, which is what
 * makes it possible to compare two embedders on identical input — without it you cannot tell
 * whether one model beats another on embedding quality or merely on alignment quality.
 */
public interface FaceEmbedder extends AutoCloseable {

	/** Aligns using the face's landmarks, then embeds. Returns an L2-normalised vector. */
	float[] embed(FaceImage image, Face face);

	/**
	 * Embeds pre-aligned pixels. Returns an L2-normalised vector.
	 *
	 * @throws UnsupportedOperationException
	 *             if the backend has no entry point that takes a crop. Some do not: InspireFace and
	 *             dlib both re-align internally from their own landmark models and expose no way
	 *             in. Ask {@link #supportsAlignedEmbed()} rather than catching this.
	 */
	float[] embed(AlignedFace aligned);

	/**
	 * Whether {@link #embed(AlignedFace)} works on this backend.
	 *
	 * <p>
	 * Worth asking, because it decides whether a comparison is meaningful. A backend that can only
	 * embed from its own alignment cannot be measured like-for-like against one that accepts a
	 * shared crop, and putting both in a single ranking silently folds alignment quality into what
	 * reads as an embedding-quality result.
	 */
	default boolean supportsAlignedEmbed() {
		return true;
	}

	/** Length of the returned vector (128 for SFace, 512 for ArcFace-family models). */
	int dimensions();

	Device device();

	@Override
	void close();
}
