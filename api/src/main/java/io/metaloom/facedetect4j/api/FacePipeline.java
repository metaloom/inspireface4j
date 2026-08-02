package io.metaloom.facedetect4j.api;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Detection and embedding together — the interface an application should normally hold.
 *
 * <p>
 * This is the seam intended to replace a direct InspireFace dependency: it names no model, no
 * inference runtime and no native library, so swapping YuNet+SFace for another pair is a change of
 * construction site rather than a change of call sites.
 */
public interface FacePipeline extends FaceDetector, FaceEmbedder {

	/**
	 * Detect every face and embed each one.
	 *
	 * <p>
	 * Faces whose landmarks are missing are returned without an embedding rather than dropped —
	 * the caller may still want the box.
	 */
	default List<Face> detectAndEmbed(FaceImage image) {
		List<Face> out = new ArrayList<>();
		for (Face f : detect(image)) {
			out.add(f.landmarks() == null ? f : f.withEmbedding(embed(image, f)));
		}
		return out;
	}

	/**
	 * How much a face at the very corner is discounted against one dead centre. At 0.5 a corner
	 * face has to be more than twice the area of a central one to win.
	 */
	double CENTER_PULL = 0.5;

	/**
	 * The face a portrait-oriented caller almost always wants: largest, with a strong pull towards
	 * the image centre.
	 *
	 * <p>
	 * The centre weighting matters — "largest" alone picks a bystander often enough to matter on
	 * real photographs. But the weighting has to be <b>relative</b> to the image, not absolute.
	 *
	 * <h2>Why not insightface's formula</h2>
	 * insightface scores {@code area - 2 * offsetFromCentreSquared}, subtracting one px² quantity
	 * from another with a fixed coefficient — so which term dominates depends entirely on the
	 * resolution. A face goes negative, and so loses to <em>any</em> more central face however
	 * small, once it is merely {@code sqrt(area/2)} from the centre: 283 px for a 400x400 face. On
	 * a 3840x2160 frame that is 7% of the half-width, so outside a small central region the size
	 * term is dead and a 60x60 bystander at the centre beats a 400x400 subject 700 px away. That is
	 * harmless for insightface, which scores already-cropped analysis-sized inputs, and wrong for
	 * the full video frames this library exists to process.
	 *
	 * <p>
	 * Here the offset is normalised against the image's own half-diagonal and applied
	 * multiplicatively, so the trade-off is identical at 640x480 and at 4K.
	 *
	 * <p>
	 * The falloff is linear in distance rather than in distance squared. Squared spends almost all
	 * of its discount on the outer third of the frame and barely separates anything nearer the
	 * middle, which is the same shape of mistake in milder form.
	 */
	default Optional<Face> primaryFace(FaceImage image, List<Face> faces) {
		double cx = image.width() / 2.0;
		double cy = image.height() / 2.0;
		double maxDist = Math.sqrt(cx * cx + cy * cy);
		return faces.stream().max(Comparator.comparingDouble(f -> {
			double dx = f.box().centerX() - cx;
			double dy = f.box().centerY() - cy;
			double offset = maxDist <= 0 ? 0 : Math.hypot(dx, dy) / maxDist;
			return f.box().area() * (1.0 - CENTER_PULL * Math.min(1.0, offset));
		}));
	}

	/** Align a detected face to the canonical 112x112 ArcFace template. */
	AlignedFace align(FaceImage image, Face face);
}
