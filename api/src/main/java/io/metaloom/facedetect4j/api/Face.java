package io.metaloom.facedetect4j.api;

import java.util.Optional;

/**
 * One detected face, optionally carrying an embedding.
 *
 * <p>
 * {@code embedding} is null until an embedder has run. It is stored L2-normalised, so
 * {@link #similarity} is a plain dot product and callers can compare vectors from different calls
 * without renormalising.
 */
public record Face(BoundingBox box, float score, Landmarks landmarks, float[] embedding) {

	public static Face of(BoundingBox box, float score, Landmarks landmarks) {
		return new Face(box, score, landmarks, null);
	}

	public Face withEmbedding(float[] emb) {
		return new Face(box, score, landmarks, emb);
	}

	public boolean hasEmbedding() {
		return embedding != null && embedding.length > 0;
	}

	public Optional<Landmarks> optionalLandmarks() {
		return Optional.ofNullable(landmarks);
	}

	/**
	 * Cosine similarity against another face's embedding, in [-1, 1].
	 *
	 * <p>
	 * Thresholds are model-specific and not transferable: SFace's useful operating point is around
	 * 0.30-0.36 while ArcFace's is around 0.25, and InspireFace ships 0.48 for Pikachu. Always
	 * calibrate against the model you actually deployed.
	 */
	public double similarity(Face other) {
		if (!hasEmbedding() || !other.hasEmbedding()
			|| embedding.length != other.embedding.length) {
			return 0;
		}
		double dot = 0;
		for (int i = 0; i < embedding.length; i++) {
			dot += (double) embedding[i] * other.embedding[i];
		}
		return dot;
	}
}
