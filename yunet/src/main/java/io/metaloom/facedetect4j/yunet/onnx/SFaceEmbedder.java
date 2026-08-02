package io.metaloom.facedetect4j.yunet.onnx;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import io.metaloom.facedetect4j.api.align.ArcFaceAlign;
import io.metaloom.facedetect4j.api.AlignedFace;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceEmbedder;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.yunet.decode.Letterbox;

/**
 * SFace face embedding on the GPU via ONNX Runtime. Apache-2.0, 128 dimensions.
 *
 * <p>
 * SFace is a MobileFaceNet trained with the sigmoid-constrained hypersphere loss. Measured on LFW
 * it reaches 99.35% verification accuracy against buffalo_l's 99.83% — about half a point behind
 * the best ArcFace model available, and ahead of InspireFace's Pikachu pack (99.16%).
 *
 * <h2>Two non-obvious things about this ONNX file</h2>
 * <ul>
 * <li><b>It declares 175 graph inputs.</b> The mxnet export left every weight tensor exposed as an
 * input rather than folding it away. All but one carry an initializer, so ONNX Runtime treats them
 * as optional — only {@code data} must be supplied. Iterating {@code getInputNames()} and trying to
 * feed them all will fail confusingly.</li>
 * <li><b>Pixels go in raw.</b> OpenCV drives it with a bare {@code blobFromImage} — no scaling, no
 * mean subtraction, no channel swap. The ArcFace-family normalisation of
 * {@code (RGB - 127.5) / 127.5} is <i>wrong</i> here and produces embeddings that still cluster,
 * still compare, and are quietly much worse.</li>
 * </ul>
 */
public class SFaceEmbedder implements FaceEmbedder {

	/** The one graph input without an initializer. */
	private static final String INPUT = "data";

	private final OrtSession session;
	private final Device device;
	private final String inputName;
	private final int dims;

	public SFaceEmbedder(Path model, Device device) {
		this.device = device;
		this.session = OrtRuntime.open(model, device);
		try {
			this.inputName = session.getInputNames().contains(INPUT)
				? INPUT
				: session.getInputNames().iterator().next();
			TensorInfo out = (TensorInfo) session.getOutputInfo().values().iterator().next().getInfo();
			long[] shape = out.getShape();
			this.dims = (int) shape[shape.length - 1];
		} catch (Exception e) {
			throw new FaceException("cannot inspect SFace model", e);
		}
	}

	@Override
	public int dimensions() {
		return dims;
	}

	@Override
	public Device device() {
		return device;
	}

	@Override
	public float[] embed(FaceImage image, Face face) {
		return embed(ArcFaceAlign.align(image, face));
	}

	@Override
	public float[] embed(AlignedFace aligned) {
		float[] blob = Letterbox.cropBlob(aligned.bgr(), AlignedFace.SIZE);
		try (OnnxTensor t = OnnxTensor.createTensor(OrtRuntime.env(), FloatBuffer.wrap(blob),
			new long[] { 1, 3, AlignedFace.SIZE, AlignedFace.SIZE })) {
			try (OrtSession.Result res = session.run(Map.of(inputName, t))) {
				FloatBuffer fb = ((OnnxTensor) res.get(0)).getFloatBuffer();
				float[] emb = new float[fb.remaining()];
				fb.get(emb);
				l2normalize(emb);
				return emb;
			}
		} catch (OrtException e) {
			throw new FaceException("SFace inference failed", e);
		}
	}

	/**
	 * Normalise in place.
	 *
	 * <p>
	 * Done here rather than left to the caller so that every vector this library hands out lives on
	 * the unit sphere. That makes cosine similarity a plain dot product and, more importantly,
	 * means vectors persisted at different times are always comparable.
	 */
	static void l2normalize(float[] v) {
		double sum = 0;
		for (float f : v) {
			sum += (double) f * f;
		}
		double n = Math.sqrt(sum);
		if (n > 0) {
			for (int i = 0; i < v.length; i++) {
				v[i] = (float) (v[i] / n);
			}
		}
	}

	@Override
	public void close() {
		try {
			session.close();
		} catch (OrtException e) {
			// nothing useful to do while tearing down
		}
	}
}
