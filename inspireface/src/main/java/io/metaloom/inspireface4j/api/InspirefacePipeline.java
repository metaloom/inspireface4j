package io.metaloom.inspireface4j.api;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.BoundingBox;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.inspireface4j.Detection;
import io.metaloom.inspireface4j.InspirefaceLib;
import io.metaloom.inspireface4j.InspirefaceSession;
import io.metaloom.inspireface4j.SessionFeature;
import io.metaloom.inspireface4j.data.FaceDetections;
import io.metaloom.opencv.core.CvType;
import io.metaloom.opencv.core.Mat;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;

/**
 * Exposes an {@link InspirefaceSession} through the backend neutral {@link FacePipeline}.
 *
 * <p>
 * Two capabilities are genuinely missing rather than merely unimplemented, and both throw
 * {@link UnsupportedOperationException} instead of being faked:
 *
 * <ul>
 * <li><b>ArcFace landmarks.</b> InspireFace's landmark model emits 106 dense points, and the
 * subset that corresponds to the ArcFace five is not documented. Guessing indices produces a crop
 * that still looks like a face and still embeds without error — it merely scores worse, with
 * nothing reporting a problem. So {@link #detect} returns faces with {@code landmarks() == null}
 * and {@link #align} refuses.</li>
 * <li><b>Embedding a supplied crop.</b> The C API aligns internally from its own landmarks and has
 * no entry point that takes pixels.</li>
 * </ul>
 *
 * <p>
 * Everything InspireFace uniquely offers — attributes, 106 landmarks, Euler angles, liveness — is
 * outside this interface by design. Reach for {@link #session()} when you want those; the point of
 * the adapter is that code which only needs boxes and vectors does not have to.
 *
 * <p>
 * <b>Not thread safe</b>, because the underlying session is not.
 */
public class InspirefacePipeline implements FacePipeline {

	private final InspirefaceSession session;
	private final boolean ownsSession;
	private int dimensions = -1;

	/**
	 * Opens a session on a model pack.
	 *
	 * @param packPath
	 *            e.g. {@code packs/Pikachu}
	 * @param detectPixelLevel
	 *            the longest-side resolution detection runs at, e.g. 640
	 */
	public static InspirefacePipeline open(String packPath, int detectPixelLevel) {
		return new InspirefacePipeline(
			InspirefaceLib.session(packPath, detectPixelLevel, SessionFeature.ENABLE_FACE_RECOGNITION),
			true);
	}

	/** Wraps a session the caller owns and will close. */
	public static InspirefacePipeline of(InspirefaceSession session) {
		return new InspirefacePipeline(session, false);
	}

	private InspirefacePipeline(InspirefaceSession session, boolean ownsSession) {
		this.session = session;
		this.ownsSession = ownsSession;
	}

	/** The wrapped session, for the InspireFace-only features the neutral API does not model. */
	public InspirefaceSession session() {
		return session;
	}

	@Override
	public List<Face> detect(FaceImage image) {
		Mat mat = toMat(image);
		try {
			return toFaces(session.detect(mat, false));
		} finally {
			MatProvider.released(mat);
		}
	}

	/**
	 * Detection and embedding in one pass — <b>the efficient path on this backend</b>.
	 *
	 * <p>
	 * InspireFace embeds by index into the native detection struct, so a {@link Face} on its own
	 * does not identify a face to the library. Doing both here, while that struct is still the
	 * session's current one, avoids the re-detection that {@link #embed(FaceImage, Face)} needs.
	 */
	@Override
	public List<Face> detectAndEmbed(FaceImage image) {
		Mat mat = toMat(image);
		try {
			FaceDetections detections = session.detect(mat, false);
			List<Face> out = new ArrayList<>(detections.size());
			for (int i = 0; i < detections.size(); i++) {
				Face face = toFace(detections.get(i));
				out.add(face.withEmbedding(normalise(session.embedding(mat, detections, i))));
			}
			return out;
		} finally {
			MatProvider.released(mat);
		}
	}

	/**
	 * Embeds a single previously detected face.
	 *
	 * <p>
	 * This re-runs detection. It has to: the native side addresses faces by index into the
	 * detection struct produced by the most recent {@code detect} call on the session, and a
	 * {@link Face} carries no such handle. Prefer {@link #detectAndEmbed} in a loop over frames.
	 */
	@Override
	public float[] embed(FaceImage image, Face face) {
		Mat mat = toMat(image);
		try {
			FaceDetections detections = session.detect(mat, false);
			int idx = bestMatch(detections, face.box());
			if (idx < 0) {
				throw new UnsupportedOperationException(
					"the face at " + face.box() + " was not found when re-detecting; it must come "
						+ "from detect() on this same image");
			}
			return normalise(session.embedding(mat, detections, idx));
		} finally {
			MatProvider.released(mat);
		}
	}

	@Override
	public float[] embed(AlignedFace aligned) {
		throw new UnsupportedOperationException(
			"InspireFace aligns internally from its own 106-point model and has no C entry point "
				+ "that accepts a crop. Use embed(FaceImage, Face), or the yunet backend when you "
				+ "need to embed pixels you aligned yourself.");
	}

	@Override
	public boolean supportsAlignedEmbed() {
		return false;
	}

	@Override
	public AlignedFace align(FaceImage image, Face face) {
		throw new UnsupportedOperationException(
			"InspireFace emits 106 dense landmarks and the mapping onto the ArcFace five is not "
				+ "documented; a guessed mapping yields crops that embed cleanly and score worse. "
				+ "Use session().landmarks(...) for the raw points.");
	}

	@Override
	public int dimensions() {
		if (dimensions < 0) {
			throw new IllegalStateException(
				"embedding size is only known after the first embed() call -- InspireFace reports "
					+ "it per feature rather than per model");
		}
		return dimensions;
	}

	/**
	 * Always {@link Device#cpu()}.
	 *
	 * <p>
	 * InspireFace runs on MNN, and its CUDA path needs the TensorRT build, which upstream
	 * documents as broken. Reporting cpu is not a fallback — it is the only mode this backend has,
	 * and a speed comparison against a GPU backend is not like-for-like.
	 */
	@Override
	public Device device() {
		return Device.cpu();
	}

	@Override
	public void close() {
		if (ownsSession) {
			try {
				session.close();
			} catch (Exception e) {
				throw new io.metaloom.facedetect4j.api.FaceException("failed to close session", e);
			}
		}
	}

	private List<Face> toFaces(FaceDetections detections) {
		List<Face> out = new ArrayList<>(detections.size());
		for (Detection d : detections) {
			out.add(toFace(d));
		}
		return out;
	}

	private static Face toFace(Detection d) {
		io.metaloom.inspireface4j.BoundingBox b = d.box();
		// Landmarks stay null on purpose -- see the class javadoc.
		return Face.of(BoundingBox.ofXYWH(b.getX(), b.getY(), b.getWidth(), b.getHeight()),
			d.conf(), null);
	}

	/** Highest-IoU detection above a loose threshold; -1 if the face is not among them. */
	private static int bestMatch(FaceDetections detections, BoundingBox target) {
		int best = -1;
		double bestIou = 0.5;
		for (int i = 0; i < detections.size(); i++) {
			double iou = target.iou(toFace(detections.get(i)).box());
			if (iou > bestIou) {
				bestIou = iou;
				best = i;
			}
		}
		return best;
	}

	/**
	 * InspireFace returns raw features. The API contract is unit length, so that
	 * {@code Face.similarity} is a plain dot product across backends.
	 */
	private float[] normalise(float[] v) {
		dimensions = v.length;
		double n = 0;
		for (float f : v) {
			n += (double) f * f;
		}
		n = Math.sqrt(n);
		if (n == 0) {
			return v;
		}
		float[] out = new float[v.length];
		for (int i = 0; i < v.length; i++) {
			out[i] = (float) (v[i] / n);
		}
		return out;
	}

	private static Mat toMat(FaceImage image) {
		BufferedImage img = new BufferedImage(image.width(), image.height(),
			BufferedImage.TYPE_3BYTE_BGR);
		// TYPE_3BYTE_BGR is byte-for-byte the layout FaceImage already holds, so this is a copy
		// rather than a conversion.
		byte[] dst = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		System.arraycopy(image.bgr(), 0, dst, 0, dst.length);

		Mat mat = MatProvider.mat(img, CvType.CV_8UC3);
		CVUtils.bufferedImageToMat(img, mat);
		return mat;
	}
}
