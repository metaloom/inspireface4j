package io.metaloom.jdlib.api;

import java.awt.Point;
import java.awt.Rectangle;
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
import io.metaloom.facedetect4j.api.Landmarks;
import io.metaloom.facedetect4j.api.align.ArcFaceAlign;
import io.metaloom.jdlib.Jdlib;
import io.metaloom.jdlib.util.FaceDescriptor;

/**
 * Exposes {@link Jdlib} through the backend neutral {@link FacePipeline}.
 *
 * <h2>What the shape predictor you chose decides</h2>
 * dlib ships two, and the choice changes what this adapter can do — as well as whether the result
 * is legally shippable:
 *
 * <table border="1">
 * <caption>predictor capabilities</caption>
 * <tr><th>Predictor</th><th>{@code detect()} landmarks</th><th>{@code align()}</th><th>Commercial use</th></tr>
 * <tr><td>{@code shape_predictor_5_face_landmarks.dat}</td><td>null</td><td>throws</td><td>yes</td></tr>
 * <tr><td>{@code shape_predictor_68_face_landmarks.dat}</td><td>ArcFace five</td><td>works</td><td><b>no</b></td></tr>
 * </table>
 *
 * <p>
 * dlib's five points are two corners of each eye plus the base of the nose — <b>no mouth
 * corners</b> — so the ArcFace five cannot be derived from them at all. That is a structural gap,
 * not a missing feature, so landmarks come back null rather than approximated. The 68-point model
 * does carry the mouth, and the iBUG numbering makes the mapping unambiguous, so it is used when
 * present. Note that this is the one model dlib states cannot be used commercially.
 *
 * <p>
 * Embedding always goes through dlib's own detect-align-embed path, which ignores any landmarks
 * supplied here, so {@link #embed(AlignedFace)} is unsupported: the JNI layer has no
 * {@code compute_face_descriptor(img, full_object_detection)} binding to route a crop through.
 *
 * <p>
 * <b>Not thread safe.</b>
 */
public class JdlibPipeline implements FacePipeline {

	/** dlib's ResNet-29 face descriptor. Fixed by the model, so it needs no probing. */
	public static final int EMBEDDING_DIMENSIONS = 128;

	/** iBUG 300-W numbering: eye corner spans, nose tip, and the two outer mouth corners. */
	private static final int LEFT_EYE_FROM = 36, LEFT_EYE_TO = 42;
	private static final int RIGHT_EYE_FROM = 42, RIGHT_EYE_TO = 48;
	private static final int NOSE_TIP = 30;
	private static final int MOUTH_LEFT = 48, MOUTH_RIGHT = 54;

	private final Jdlib jdlib;

	/**
	 * @param jdlib
	 *            a configured instance. Construct it with the embedding model too if you intend to
	 *            call {@link #embed}; {@link Jdlib} only reports that omission when you do.
	 */
	public JdlibPipeline(Jdlib jdlib) {
		this.jdlib = jdlib;
	}

	/** The wrapped instance, for the CNN detector and the 68-point landmarks in full. */
	public Jdlib jdlib() {
		return jdlib;
	}

	/**
	 * HOG detection, with landmarks when the loaded predictor can supply them.
	 *
	 * <p>
	 * This makes two native passes, because dlib's box and landmark entry points are separate. The
	 * landmark pass re-detects internally, so its faces are matched back onto the first pass's
	 * boxes by IoU rather than by position.
	 */
	@Override
	public List<Face> detect(FaceImage image) {
		BufferedImage img = toBufferedImage(image);
		List<Rectangle> boxes = jdlib.detectFace(img);
		if (boxes.isEmpty()) {
			return List.of();
		}

		List<FaceDescriptor> shapes = safeLandmarks(img);
		List<Face> out = new ArrayList<>(boxes.size());
		for (Rectangle r : boxes) {
			BoundingBox box = toBox(r);
			out.add(Face.of(box, DETECTION_SCORE, landmarksFor(box, shapes)));
		}
		return out;
	}

	/**
	 * dlib's HOG binding reports no confidence, so every detection scores 1.0.
	 *
	 * <p>
	 * Stated rather than hidden: a score-based filter or a score-weighted ranking is meaningless
	 * against this backend, and it will look like it is working.
	 */
	public static final float DETECTION_SCORE = 1.0f;

	@Override
	public float[] embed(FaceImage image, Face face) {
		BufferedImage img = toBufferedImage(image);
		for (FaceDescriptor d : jdlib.getFaceEmbeddings(img)) {
			if (d.getFaceEmbedding() != null && toBox(d.getFaceBox()).iou(face.box()) > 0.5) {
				return normalise(d.getFaceEmbedding());
			}
		}
		throw new io.metaloom.facedetect4j.api.FaceException(
			"dlib produced no embedding for the face at " + face.box()
				+ " -- it re-detects internally, so the face must be one dlib itself finds");
	}

	/**
	 * All faces in one native pass, which is the efficient path here: dlib's embedding entry point
	 * detects, aligns and embeds every face at once, so calling {@link #embed} per face would
	 * repeat the whole thing.
	 */
	@Override
	public List<Face> detectAndEmbed(FaceImage image) {
		BufferedImage img = toBufferedImage(image);
		List<FaceDescriptor> descriptors = jdlib.getFaceEmbeddings(img);
		List<Face> out = new ArrayList<>(descriptors.size());
		for (FaceDescriptor d : descriptors) {
			BoundingBox box = toBox(d.getFaceBox());
			Face face = Face.of(box, DETECTION_SCORE, toArcFaceFive(d.getFacialLandmarks()));
			out.add(d.getFaceEmbedding() == null ? face
				: face.withEmbedding(normalise(d.getFaceEmbedding())));
		}
		return out;
	}

	@Override
	public float[] embed(AlignedFace aligned) {
		throw new UnsupportedOperationException(
			"jdlib has no compute_face_descriptor(img, full_object_detection) binding, so a crop "
				+ "cannot be routed into dlib's embedder. Use embed(FaceImage, Face), or the yunet "
				+ "backend when you need to embed pixels you aligned yourself.");
	}

	@Override
	public boolean supportsAlignedEmbed() {
		return false;
	}

	/**
	 * Aligns to the shared ArcFace template — <b>note this is not what {@link #embed} uses</b>.
	 * dlib re-aligns internally with its own 5-point chip extraction, so a crop from here is for
	 * feeding some other embedder, not this one.
	 */
	@Override
	public AlignedFace align(FaceImage image, Face face) {
		if (face.landmarks() == null) {
			throw new UnsupportedOperationException(
				"no ArcFace landmarks for this face. dlib's 5-point predictor gives eye corners "
					+ "and the nose base but no mouth corners, so the ArcFace five cannot be "
					+ "derived from it; load shape_predictor_68_face_landmarks.dat if you need "
					+ "alignment (and check its licence -- it is not free for commercial use)");
		}
		return ArcFaceAlign.align(image, face);
	}

	@Override
	public int dimensions() {
		return EMBEDDING_DIMENSIONS;
	}

	/**
	 * Always {@link Device#cpu()}.
	 *
	 * <p>
	 * The HOG detector, the shape predictor and the embedder are all CPU. dlib's CNN detector can
	 * use CUDA when the native library was built against it, but it is reached through
	 * {@link Jdlib#cnnDetectFace} rather than this interface, and this adapter does not claim GPU
	 * on its behalf.
	 */
	@Override
	public Device device() {
		return Device.cpu();
	}

	@Override
	public void close() {
		// Jdlib holds model handles created per call on the native side and exposes no release.
	}

	private List<FaceDescriptor> safeLandmarks(BufferedImage img) {
		try {
			return jdlib.getFaceLandmarks(img);
		} catch (IllegalArgumentException e) {
			// No predictor configured. Boxes are still useful, so this degrades rather than fails.
			return List.of();
		}
	}

	private static Landmarks landmarksFor(BoundingBox box, List<FaceDescriptor> shapes) {
		for (FaceDescriptor d : shapes) {
			if (toBox(d.getFaceBox()).iou(box) > 0.5) {
				return toArcFaceFive(d.getFacialLandmarks());
			}
		}
		return null;
	}

	/**
	 * Maps dlib's landmark output onto the ArcFace five, or null when it cannot be done.
	 *
	 * <p>
	 * Only the 68-point layout carries mouth corners. Anything else returns null rather than an
	 * approximation, because a subtly wrong template produces a crop that still embeds without
	 * error and merely scores worse.
	 */
	static Landmarks toArcFaceFive(List<Point> points) {
		if (points == null || points.size() != 68) {
			return null;
		}
		float[] xy = new float[10];
		centroid(points, LEFT_EYE_FROM, LEFT_EYE_TO, xy, 0);
		centroid(points, RIGHT_EYE_FROM, RIGHT_EYE_TO, xy, 1);
		put(points.get(NOSE_TIP), xy, 2);
		put(points.get(MOUTH_LEFT), xy, 3);
		put(points.get(MOUTH_RIGHT), xy, 4);
		return new Landmarks(xy);
	}

	private static void centroid(List<Point> pts, int from, int to, float[] xy, int slot) {
		double sx = 0, sy = 0;
		for (int i = from; i < to; i++) {
			sx += pts.get(i).x;
			sy += pts.get(i).y;
		}
		int n = to - from;
		xy[2 * slot] = (float) (sx / n);
		xy[2 * slot + 1] = (float) (sy / n);
	}

	private static void put(Point p, float[] xy, int slot) {
		xy[2 * slot] = p.x;
		xy[2 * slot + 1] = p.y;
	}

	private static BoundingBox toBox(Rectangle r) {
		return BoundingBox.ofXYWH(r.x, r.y, r.width, r.height);
	}

	/** dlib's descriptors are not unit length; the API contract is that they are. */
	private static float[] normalise(float[] v) {
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

	/**
	 * dlib returns nothing at all from a {@code TYPE_INT_RGB} image, silently — its JNI layer
	 * reads the raster bytes directly and only {@code TYPE_3BYTE_BGR} has the layout it expects.
	 */
	private static BufferedImage toBufferedImage(FaceImage image) {
		BufferedImage img = new BufferedImage(image.width(), image.height(),
			BufferedImage.TYPE_3BYTE_BGR);
		byte[] dst = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		System.arraycopy(image.bgr(), 0, dst, 0, dst.length);
		return img;
	}
}
