package io.metaloom.facedetect4j.yunet.decode;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import io.metaloom.facedetect4j.api.BoundingBox;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.Landmarks;

/**
 * Decodes YuNet's twelve output heads into faces.
 *
 * <p>
 * A direct port of OpenCV's {@code FaceDetectorYNImpl::postProcess}, which is the only normative
 * description of this model's output that exists — YuNet ships no decoding spec.
 *
 * <h2>Output layout</h2>
 * Twelve tensors, four families across three strides:
 *
 * <pre>
 *   cls_{8,16,32}   [1, N, 1]    classification logit-ish, already in [0,1]
 *   obj_{8,16,32}   [1, N, 1]    objectness, already in [0,1]
 *   bbox_{8,16,32}  [1, N, 4]    cx, cy offsets in cell units; w, h in log space
 *   kps_{8,16,32}   [1, N, 10]   five points, offsets in cell units
 * </pre>
 *
 * with {@code N = (H/stride) * (W/stride)} and — unlike SCRFD — <b>one anchor per cell</b>.
 *
 * <h2>Two things that are easy to get wrong</h2>
 * <ul>
 * <li><b>The score is {@code sqrt(cls * obj)}</b>, not either one alone and not their product.
 * Using {@code cls} by itself shifts the whole score distribution and silently changes where a
 * threshold sits.</li>
 * <li><b>Width and height are exponentiated, the centre is not.</b> {@code cx} and {@code cy} are
 * plain offsets added to the cell index; {@code w} and {@code h} are {@code exp()} of the
 * prediction. Applying {@code exp} to all four yields boxes that are plausibly placed and wrongly
 * sized.</li>
 * </ul>
 */
public final class YuNetDecoder {

	public static final int[] STRIDES = { 8, 16, 32 };

	private YuNetDecoder() {
	}

	/** One model output: flat data plus shape. */
	public record Head(String name, float[] data) {
	}

	/**
	 * @param cls     three cls heads, ordered by stride 8, 16, 32
	 * @param obj     three obj heads, same order
	 * @param bbox    three bbox heads, same order
	 * @param kps     three kps heads, same order
	 * @param netW    network input width (after letterboxing)
	 * @param netH    network input height
	 * @param scale   factor the source image was multiplied by; results are divided back out
	 * @param scoreThresh keep faces at or above this
	 * @param nmsThresh   IoU above which a lower-scoring box is suppressed
	 * @param topK    keep at most this many after NMS ({@code <= 0} for unlimited)
	 */
	public static List<Face> decode(float[][] cls, float[][] obj, float[][] bbox, float[][] kps,
		int netW, int netH, float scale, float scoreThresh, float nmsThresh, int topK) {

		List<Face> pre = new ArrayList<>();

		for (int s = 0; s < STRIDES.length; s++) {
			int stride = STRIDES[s];
			int cols = netW / stride;
			int rows = netH / stride;
			float[] c = cls[s], o = obj[s], b = bbox[s], k = kps[s];

			for (int r = 0; r < rows; r++) {
				for (int col = 0; col < cols; col++) {
					int idx = r * cols + col;
					if (idx >= c.length) {
						continue;
					}
					float clsScore = clamp01(c[idx]);
					float objScore = clamp01(o[idx]);
					float score = (float) Math.sqrt(clsScore * objScore);
					if (score < scoreThresh) {
						continue;
					}

					int b4 = idx * 4;
					float cx = (col + b[b4]) * stride;
					float cy = (r + b[b4 + 1]) * stride;
					float w = (float) Math.exp(b[b4 + 2]) * stride;
					float h = (float) Math.exp(b[b4 + 3]) * stride;

					float[] pts = new float[10];
					int k10 = idx * 10;
					for (int p = 0; p < 5; p++) {
						pts[2 * p] = (col + k[k10 + 2 * p]) * stride;
						pts[2 * p + 1] = (r + k[k10 + 2 * p + 1]) * stride;
					}

					pre.add(Face.of(
						new BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
						score, new Landmarks(pts)));
				}
			}
		}

		pre.sort(Comparator.comparingDouble(Face::score).reversed());
		List<Face> kept = Nms.greedy(pre, nmsThresh);
		if (topK > 0 && kept.size() > topK) {
			kept = kept.subList(0, topK);
		}

		// Undo the letterbox scale so coordinates are in source-image pixels.
		float inv = 1f / scale;
		List<Face> out = new ArrayList<>(kept.size());
		for (Face f : kept) {
			out.add(new Face(f.box().scaled(inv), f.score(),
				f.landmarks() == null ? null : f.landmarks().scaled(inv), null));
		}
		return out;
	}

	private static float clamp01(float v) {
		return v < 0f ? 0f : (v > 1f ? 1f : v);
	}
}
