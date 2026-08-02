package io.metaloom.facedetect4j.yunet.onnx;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import io.metaloom.facedetect4j.api.Device;
import io.metaloom.facedetect4j.api.Face;
import io.metaloom.facedetect4j.api.FaceDetector;
import io.metaloom.facedetect4j.api.FaceException;
import io.metaloom.facedetect4j.api.FaceImage;
import io.metaloom.facedetect4j.yunet.decode.Letterbox;
import io.metaloom.facedetect4j.yunet.decode.YuNetDecoder;

/**
 * YuNet face detection on the GPU via ONNX Runtime.
 *
 * <p>
 * YuNet is MIT-licensed, roughly 230 KB, and detects faces from about 10x10 pixels upward. On
 * WIDER FACE val it measures AP 0.884 / 0.867 / 0.753 across easy / medium / hard — notably
 * <i>better on the hard subset</i> than SCRFD-500MF (0.674) and SCRFD-2.5GF (0.750), which is
 * where small faces live.
 *
 * <h2>Two model files, two behaviours</h2>
 * <ul>
 * <li>{@code face_detection_yunet_2026may.onnx} has a <b>dynamic</b> input shape, so the network
 * runs at the image's own resolution. Preferred: no letterbox padding, better on small faces.</li>
 * <li>{@code face_detection_yunet_2023mar.onnx} is fixed at <b>640x640</b>. ONNX Runtime cannot
 * reshape a static graph the way OpenCV's DNN module can, so this detector letterboxes into 640
 * when it sees that model.</li>
 * </ul>
 * Which one is loaded is detected from the graph, not configured, so passing either file just
 * works.
 */
public class YuNetDetector implements FaceDetector {

	/**
	 * Input dimensions are rounded up to a multiple of this. <b>Mandatory, not cosmetic.</b>
	 *
	 * <p>
	 * YuNet's neck adds an upsampled stride-32 feature map to the stride-16 one. When the input is
	 * not 32-divisible those two paths round differently and the addition cannot broadcast, so ONNX
	 * Runtime aborts mid-graph. Measured directly on {@code face_detection_yunet_2026may.onnx}:
	 *
	 * <pre>
	 *   704x704  -> OK          677x687 -> FAIL: "Add_44 ... Attempting to broadcast an axis by a
	 *   704x992  -> OK                            dimension other than 1. 42 by 43"
	 * </pre>
	 *
	 * OpenCV's DNN engine tolerates the ragged sizes; ONNX Runtime does not. So a port from
	 * {@code FaceDetectorYN} that simply forwards {@code setInputSize(w, h)} will work on some
	 * images and throw on others, with the failure depending on nothing more than image dimensions.
	 *
	 * <p>
	 * The cost is a few pixels of box drift versus OpenCV on non-aligned images (measured: 0.06 px
	 * on 677x687, 4.4 px on 675x963), because the padding slightly changes the effective geometry.
	 * At 32-aligned sizes the grids use floor division, which is what {@link YuNetDecoder} assumes.
	 */
	private static final int STRIDE_ALIGN = 32;

	private final OrtSession session;
	private final Device device;
	private final String inputName;
	private final boolean dynamic;
	private final int fixedW;
	private final int fixedH;

	private float scoreThreshold = 0.6f;
	private float nmsThreshold = 0.3f;
	private int topK = 5000;
	private int maxInputEdge = 1280;

	public YuNetDetector(Path model, Device device) {
		this.device = device;
		this.session = OrtRuntime.open(model, device);
		try {
			this.inputName = session.getInputNames().iterator().next();
			TensorInfo info = (TensorInfo) session.getInputInfo().get(inputName).getInfo();
			long[] shape = info.getShape();
			// A negative or zero dim is ONNX's way of saying "symbolic".
			this.dynamic = shape[2] <= 0 || shape[3] <= 0;
			this.fixedH = dynamic ? -1 : (int) shape[2];
			this.fixedW = dynamic ? -1 : (int) shape[3];
		} catch (Exception e) {
			throw new FaceException("cannot inspect YuNet input shape", e);
		}
	}

	/** Score floor. YuNet's distribution is flatter than SCRFD's; 0.6-0.9 is the useful band. */
	public YuNetDetector setScoreThreshold(float v) {
		this.scoreThreshold = v;
		return this;
	}

	public YuNetDetector setNmsThreshold(float v) {
		this.nmsThreshold = v;
		return this;
	}

	public YuNetDetector setTopK(int v) {
		this.topK = v;
		return this;
	}

	/**
	 * Cap on the longer edge for the dynamic model. Bounds VRAM and latency on very large images;
	 * raise it to find smaller faces in high-resolution stills.
	 */
	public YuNetDetector setMaxInputEdge(int v) {
		this.maxInputEdge = v;
		return this;
	}

	@Override
	public Device device() {
		return device;
	}

	@Override
	public List<Face> detect(FaceImage image) {
		int netW, netH;
		float scale;
		if (dynamic) {
			// Run at native resolution, downscaling only if it exceeds the cap, then round up to
			// a multiple of 32 so every stride head divides evenly.
			int longEdge = Math.max(image.width(), image.height());
			scale = longEdge > maxInputEdge ? (float) maxInputEdge / longEdge : 1f;
			netW = align(Math.round(image.width() * scale));
			netH = align(Math.round(image.height() * scale));
		} else {
			netW = fixedW;
			netH = fixedH;
			scale = Letterbox.scaleFor(image.width(), image.height(), netW, netH);
		}

		float[] blob = Letterbox.blob(image, netW, netH, scale);
		try (OnnxTensor t = OnnxTensor.createTensor(OrtRuntime.env(), FloatBuffer.wrap(blob),
			new long[] { 1, 3, netH, netW })) {
			try (OrtSession.Result res = session.run(Map.of(inputName, t))) {
				float[][] cls = new float[3][];
				float[][] obj = new float[3][];
				float[][] bbox = new float[3][];
				float[][] kps = new float[3][];
				for (int i = 0; i < YuNetDecoder.STRIDES.length; i++) {
					int s = YuNetDecoder.STRIDES[i];
					cls[i] = out(res, "cls_" + s);
					obj[i] = out(res, "obj_" + s);
					bbox[i] = out(res, "bbox_" + s);
					kps[i] = out(res, "kps_" + s);
				}
				return YuNetDecoder.decode(cls, obj, bbox, kps, netW, netH, scale,
					scoreThreshold, nmsThreshold, topK);
			}
		} catch (OrtException e) {
			throw new FaceException("YuNet inference failed", e);
		}
	}

	private static int align(int v) {
		int a = ((v + STRIDE_ALIGN - 1) / STRIDE_ALIGN) * STRIDE_ALIGN;
		return Math.max(STRIDE_ALIGN, a);
	}

	private static float[] out(OrtSession.Result res, String name) throws OrtException {
		var v = res.get(name).orElseThrow(
			() -> new FaceException("YuNet output '" + name + "' missing -- is this actually a "
				+ "YuNet model? Expected cls_/obj_/bbox_/kps_ heads at strides 8, 16 and 32."));
		FloatBuffer fb = ((OnnxTensor) v).getFloatBuffer();
		float[] a = new float[fb.remaining()];
		fb.get(a);
		return a;
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
