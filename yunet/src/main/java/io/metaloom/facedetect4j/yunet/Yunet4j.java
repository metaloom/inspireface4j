package io.metaloom.facedetect4j.yunet;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import io.metaloom.facedetect4j.api.align.ArcFaceAlign;
import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.api.FacePipeline;
import io.metaloom.facedetect4j.yunet.onnx.SFaceEmbedder;
import io.metaloom.facedetect4j.yunet.onnx.YuNetDetector;

/**
 * Entry point. Builds a GPU {@link FacePipeline} from a directory of models.
 *
 * <pre>{@code
 * try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"))) {
 *     FaceImage img = FaceImage.read(Path.of("photo.jpg"));
 *     for (Face f : faces.detectAndEmbed(img)) {
 *         System.out.println(f.box() + " score=" + f.score());
 *     }
 * }
 * }</pre>
 *
 * <p>
 * The models are not bundled: YuNet and SFace are MIT and Apache-2.0 respectively, but they are
 * distributed through opencv_zoo's git-lfs storage and are better fetched than vendored. See
 * {@link #downloadHint()}.
 */
public final class Yunet4j {

	/** Preferred detector: dynamic input shape, so inference runs at native resolution. */
	public static final String YUNET_DYNAMIC = "face_detection_yunet_2026may.onnx";
	/** Fallback detector: fixed 640x640, letterboxed. */
	public static final String YUNET_FIXED = "face_detection_yunet_2023mar.onnx";
	public static final String SFACE = "face_recognition_sface_2021dec.onnx";

	private Yunet4j() {
	}

	/** GPU 0. Throws if CUDA is unavailable — this library does not fall back silently. */
	public static FacePipeline pipeline(Path modelDir) {
		return pipeline(modelDir, Device.cuda());
	}

	public static FacePipeline pipeline(Path modelDir, Device device) {
		Path det = resolveDetector(modelDir);
		Path rec = modelDir.resolve(SFACE);
		if (!Files.isRegularFile(rec)) {
			throw new FaceException("SFace model not found: " + rec + "\n" + downloadHint());
		}
		return new YuNetSFacePipeline(new YuNetDetector(det, device),
			new SFaceEmbedder(rec, device));
	}

	/** Prefer the dynamic export; fall back to the fixed-shape one if that is what is present. */
	static Path resolveDetector(Path modelDir) {
		Path dyn = modelDir.resolve(YUNET_DYNAMIC);
		if (Files.isRegularFile(dyn)) {
			return dyn;
		}
		Path fixed = modelDir.resolve(YUNET_FIXED);
		if (Files.isRegularFile(fixed)) {
			return fixed;
		}
		throw new FaceException("no YuNet model in " + modelDir + " (looked for "
			+ YUNET_DYNAMIC + " and " + YUNET_FIXED + ")\n" + downloadHint());
	}

	/**
	 * How to fetch the models.
	 *
	 * <p>
	 * The {@code media.} host matters: opencv_zoo stores these under git-lfs, and
	 * {@code raw.githubusercontent.com} serves a ~130-byte text pointer with a {@code .onnx} name
	 * instead of the model. That failure surfaces much later as an opaque protobuf error.
	 */
	public static String downloadHint() {
		String base = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models";
		return """
			Fetch the models (note the media. host -- opencv_zoo is git-lfs):
			  curl -L -o %s/%s \\
			    %s/face_detection_yunet/%s
			  curl -L -o %s/%s \\
			    %s/face_recognition_sface/%s
			""".formatted("<modelDir>", YUNET_DYNAMIC, base, YUNET_DYNAMIC,
			"<modelDir>", SFACE, base, SFACE);
	}

	/** Detector plus embedder behind one handle. */
	static final class YuNetSFacePipeline implements FacePipeline {

		private final YuNetDetector detector;
		private final SFaceEmbedder embedder;

		YuNetSFacePipeline(YuNetDetector detector, SFaceEmbedder embedder) {
			this.detector = detector;
			this.embedder = embedder;
		}

		@Override
		public List<Face> detect(FaceImage image) {
			return detector.detect(image);
		}

		@Override
		public float[] embed(FaceImage image, Face face) {
			return embedder.embed(image, face);
		}

		@Override
		public float[] embed(AlignedFace aligned) {
			return embedder.embed(aligned);
		}

		@Override
		public AlignedFace align(FaceImage image, Face face) {
			return ArcFaceAlign.align(image, face);
		}

		@Override
		public int dimensions() {
			return embedder.dimensions();
		}

		@Override
		public Device device() {
			return detector.device();
		}

		/** Exposed so callers can tune thresholds without unwrapping the pipeline abstraction. */
		public YuNetDetector detector() {
			return detector;
		}

		@Override
		public void close() {
			detector.close();
			embedder.close();
		}
	}
}
