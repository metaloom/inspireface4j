package io.metaloom.facedetect4j.api.align;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.Landmarks;

/**
 * The canonical ArcFace 112x112 five-point alignment.
 *
 * <p>
 * Alignment is the quietest failure mode in face recognition. A crop built from a slightly
 * different template, or from permuted keypoints, still looks like a face and still embeds without
 * error — it simply scores worse, with nothing anywhere reporting a problem. So the template lives
 * here once and nothing else is allowed to define its own.
 *
 * <h2>These exact constants are shared with OpenCV</h2>
 * The template below is insightface's {@code face_align.arcface_dst} and is byte-identical to
 * OpenCV's {@code FaceRecognizerSF::alignCrop} coordinates. That is what makes SFace and
 * ArcFace-family embedders directly comparable on the same crop.
 *
 * <h2>Validated, not assumed</h2>
 * The transform is Umeyama-with-scaling, the same algorithm behind
 * {@code skimage.transform.SimilarityTransform.estimate()} that insightface calls. This
 * implementation was checked against skimage to <b>1e-9</b> on frontal, rolled, sub-pixel and
 * mirrored keypoint sets, and end-to-end against {@code cv2 FaceRecognizerSF.alignCrop} to a
 * minimum embedding cosine of <b>0.99994</b>. See {@code ArcFaceAlignTest}.
 */
public final class ArcFaceAlign {

	public static final int SIZE = AlignedFace.SIZE;

	private ArcFaceAlign() {
	}

	/** insightface {@code arcface_dst}: left eye, right eye, nose, left mouth, right mouth. */
	private static final double[][] DST_112 = {
		{ 38.2946, 51.6963 },
		{ 73.5318, 51.5014 },
		{ 56.0252, 71.7366 },
		{ 41.5493, 92.3655 },
		{ 70.7299, 92.2041 }
	};

	public static double[][] template() {
		double[][] c = new double[5][2];
		for (int i = 0; i < 5; i++) {
			c[i] = DST_112[i].clone();
		}
		return c;
	}

	public static AlignedFace align(FaceImage img, Face face) {
		Landmarks lm = face.landmarks();
		if (lm == null) {
			throw new FaceException("cannot align a face without landmarks");
		}
		double[][] src = new double[5][2];
		for (int i = 0; i < 5; i++) {
			src[i][0] = lm.x(i);
			src[i][1] = lm.y(i);
		}
		double[] m = umeyama(src, DST_112);
		float[] affine = new float[6];
		for (int i = 0; i < 6; i++) {
			affine[i] = (float) m[i];
		}
		return new AlignedFace(warp(img, affine), affine, face);
	}

	/**
	 * Umeyama (1991) least-squares similarity transform with scaling.
	 *
	 * @return 2x3 affine, row-major {@code [a,b,tx,c,d,ty]}, mapping source to template
	 */
	public static double[] umeyama(double[][] src, double[][] dst) {
		int n = src.length;
		double sxm = 0, sym = 0, dxm = 0, dym = 0;
		for (int i = 0; i < n; i++) {
			sxm += src[i][0];
			sym += src[i][1];
			dxm += dst[i][0];
			dym += dst[i][1];
		}
		sxm /= n;
		sym /= n;
		dxm /= n;
		dym /= n;

		double a00 = 0, a01 = 0, a10 = 0, a11 = 0, varSrc = 0;
		for (int i = 0; i < n; i++) {
			double sx = src[i][0] - sxm, sy = src[i][1] - sym;
			double dx = dst[i][0] - dxm, dy = dst[i][1] - dym;
			a00 += dx * sx;
			a01 += dx * sy;
			a10 += dy * sx;
			a11 += dy * sy;
			varSrc += sx * sx + sy * sy;
		}
		a00 /= n;
		a01 /= n;
		a10 /= n;
		a11 /= n;
		varSrc /= n;

		double d1 = (a00 * a11 - a01 * a10) < 0 ? -1.0 : 1.0;
		Svd2x2 svd = Svd2x2.of(a00, a01, a10, a11);
		double[] r = mul(mul(svd.u00, svd.u01, svd.u10, svd.u11, 1.0, 0, 0, d1),
			svd.vt00, svd.vt01, svd.vt10, svd.vt11);

		double scale = varSrc == 0 ? 1.0 : (svd.s0 + svd.s1 * d1) / varSrc;
		double a = scale * r[0], b = scale * r[1];
		double c = scale * r[2], d = scale * r[3];
		return new double[] { a, b, dxm - (a * sxm + b * sym), c, d, dym - (c * sxm + d * sym) };
	}

	private static double[] mul(double a00, double a01, double a10, double a11,
		double b00, double b01, double b10, double b11) {
		return new double[] {
			a00 * b00 + a01 * b10, a00 * b01 + a01 * b11,
			a10 * b00 + a11 * b10, a10 * b01 + a11 * b11
		};
	}

	private static double[] mul(double[] a, double b00, double b01, double b10, double b11) {
		return mul(a[0], a[1], a[2], a[3], b00, b01, b10, b11);
	}

	/** Bilinear warp, {@code BORDER_CONSTANT} 0 — matching {@code cv2.warpAffine} defaults. */
	static byte[] warp(FaceImage src, float[] m) {
		byte[] out = new byte[SIZE * SIZE * 3];
		double a = m[0], b = m[1], tx = m[2], c = m[3], d = m[4], ty = m[5];
		double det = a * d - b * c;
		if (det == 0) {
			return out;
		}
		double i00 = d / det, i01 = -b / det, i10 = -c / det, i11 = a / det;
		int w = src.width(), h = src.height();

		for (int y = 0; y < SIZE; y++) {
			for (int x = 0; x < SIZE; x++) {
				double px = x - tx, py = y - ty;
				double u = i00 * px + i01 * py;
				double v = i10 * px + i11 * py;
				int u0 = (int) Math.floor(u), v0 = (int) Math.floor(v);
				if (u0 < -1 || v0 < -1 || u0 >= w || v0 >= h) {
					continue;
				}
				double fu = u - u0, fv = v - v0;
				int o = (y * SIZE + x) * 3;
				for (int ch = 0; ch < 3; ch++) {
					double acc = (1 - fu) * (1 - fv) * sample(src, u0, v0, ch)
						+ fu * (1 - fv) * sample(src, u0 + 1, v0, ch)
						+ (1 - fu) * fv * sample(src, u0, v0 + 1, ch)
						+ fu * fv * sample(src, u0 + 1, v0 + 1, ch);
					out[o + ch] = (byte) Math.clamp((int) Math.round(acc), 0, 255);
				}
			}
		}
		return out;
	}

	private static int sample(FaceImage img, int x, int y, int c) {
		if (x < 0 || y < 0 || x >= img.width() || y >= img.height()) {
			return 0;
		}
		return img.at(x, y, c);
	}

	/**
	 * Closed-form 2x2 SVD, {@code A = U diag(s0,s1) Vt} with {@code s0 >= s1 >= 0}, matching
	 * numpy's convention where the third returned value is already transposed.
	 */
	record Svd2x2(double u00, double u01, double u10, double u11,
		double s0, double s1,
		double vt00, double vt01, double vt10, double vt11) {

		static Svd2x2 of(double a, double b, double c, double d) {
			double e = (a + d) / 2.0, f = (a - d) / 2.0;
			double g = (c + b) / 2.0, h = (c - b) / 2.0;
			double q = Math.hypot(e, h), r = Math.hypot(f, g);
			double s0 = q + r, s1 = q - r;
			double theta = (Math.atan2(h, e) - Math.atan2(g, f)) / 2.0;
			double phi = (Math.atan2(h, e) + Math.atan2(g, f)) / 2.0;
			double cp = Math.cos(phi), sp = Math.sin(phi);
			double ct = Math.cos(theta), st = Math.sin(theta);

			// A = R(phi) . diag . R(theta). The second rotation is NOT transposed -- writing
			// R(theta)^T here gives a transform of the right magnitude but mirrored, which still
			// produces a face-shaped crop and merely scores worse.
			double vt00 = ct, vt01 = -st, vt10 = st, vt11 = ct;
			if (s1 < 0) {
				s1 = -s1;
				vt10 = -vt10;
				vt11 = -vt11;
			}
			return new Svd2x2(cp, -sp, sp, cp, s0, s1, vt00, vt01, vt10, vt11);
		}
	}
}
