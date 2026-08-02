# Facedetect4J YuNet

GPU face **detection** (YuNet) and **embedding** (SFace) for Java, on ONNX Runtime.

The permissively licensed backend of [Facedetect4J](../): both models carry explicit commercial
grants — YuNet is **MIT**, SFace is **Apache-2.0** — where the InspireFace and InsightFace model
packs are stated by their authors to be for academic use only.

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-yunet</artifactId>
  <version>0.1.0-SNAPSHOT</version>
</dependency>
```

## Why this exists

Measured over 6000 LFW pairs and 3226 WIDER FACE val images:

| | LFW accuracy | TAR@FAR=1e-3 | WIDER AP easy/med/hard | Licence |
|---|---:|---:|---|---|
| **YuNet + SFace** | **99.35 %** | 98.57 % | 0.884 / 0.867 / **0.753** | 🟡 MIT + Apache-2.0 |
| InspireFace Pikachu | 99.16 % | 97.59 % | — | 🔴 non-commercial |
| InspireFace Megatron | 99.58 % | 99.17 % | — | 🔴 non-commercial |
| insightface buffalo_l | 99.83 % | 99.70 % | 0.948 / 0.933 / 0.808 | 🔴 non-commercial |

Against the InspireFace pack MetaLoom ships today (Pikachu) this is **more accurate**, better on the
hard WIDER subset than any SCRFD variant below 10GF, and legally shippable. Against the ceiling
(buffalo_l) it gives up about half a point of LFW accuracy.

## Usage

Detection:

```java
// GPU 0. Throws if the CUDA provider will not attach - it never falls back silently.
try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"))) {

	FaceImage img = FaceImage.read(photo);

	// Every face above threshold, unsorted and uncapped
	for (Face face : faces.detect(img)) {
		System.out.println(face.box() + " @ " + String.format("%.4f", face.score()));
	}
}
```

Embedding:

```java
try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"), Device.cuda())) {

	FaceImage img = FaceImage.read(photo);

	// Detection and embedding in one pass
	for (Face face : faces.detectAndEmbed(img)) {
		System.out.println(face.box() + " -> " + face.embedding().length + "d");
	}

	// Or pick the single face a portrait pipeline wants
	Face a = faces.primaryFace(img, faces.detect(img)).orElseThrow();
	a = a.withEmbedding(faces.embed(img, a));

	FaceImage other = FaceImage.read(otherPhoto);
	Face b = faces.primaryFace(other, faces.detect(other)).orElseThrow();
	b = b.withEmbedding(faces.embed(other, b));

	// Embeddings are L2 normalised, so comparing is a plain dot product:
	// ~0.65 for the same person, ~0.16 for different people
	System.out.println("similarity: " + a.similarity(b));

	// Align once, embed from the crop - the entry point that lets two embedders be
	// compared on identical pixels
	AlignedFace crop = faces.align(img, a);
	float[] embedding = faces.embed(crop);
	System.out.println("dimensions: " + embedding.length);
}
```

Feeding pixels from an existing decoder skips AWT entirely:

```java
FaceImage img = FaceImage.ofBgrBytes(width, height, bgrBytes);
```

Video, via [video4j](https://github.com/metaloom/video4j) — detections printed and drawn onto the
frame, in a viewer window:

```java
Video4j.init();
SimpleImageViewer viewer = new SimpleImageViewer();

Face reference = null;
List<Double> identity = new ArrayList<>();

try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"));
	VideoFile video = VideoFile.open(clip)) {

	VideoFrame decoded;
	while ((decoded = video.frame()) != null) {

		// try-with-resources releases the frame's Mat. video4j allocates a fresh one per
		// frame and native memory is invisible to the garbage collector, so a long clip
		// processed without this grows until the OS intervenes rather than the JVM.
		try (VideoFrame frame = decoded) {

			// No resize step: the detector caps the long edge itself
			// (YuNetDetector.setMaxInputEdge), so one here would only add a second
			// interpolation and move the boxes.
			FaceImage img = toFaceImage(frame.mat());

			List<Face> detections = faces.detect(img);

			// Print the detections, and draw them onto the frame the viewer shows.
			// Drawing after the conversion above, so the boxes never reach the model.
			for (Face detection : detections) {
				BoundingBox box = detection.box();
				System.out.println("Frame[" + video.currentFrame() + "] = "
					+ detection.score() + " @ " + box);
				CVUtils.drawRect(frame.mat(), (int) box.x1(), (int) box.y1(),
					(int) box.width(), (int) box.height(), new Scalar(0, 255, 0));
			}

			// The single face a portrait pipeline wants, embedded and compared against
			// the first frame's. Embeddings from separate calls are directly comparable,
			// so "is this still the same person" needs no state beyond one reference
			// vector — no tracker, no frame-to-frame association.
			Optional<Face> found = faces.primaryFace(img, detections);
			if (found.isPresent()) {
				Face face = found.get().withEmbedding(faces.embed(img, found.get()));
				if (reference == null) {
					reference = face;
				}
				identity.add(reference.similarity(face));
			}

			viewer.show(frame.mat());
		}
	}
}

// Judge a clip on the distribution, not on the worst frame. This one is a head turning
// through full profile, and at profile SFace sees one eye and no mouth corners: the
// cosine against a frontal reference falls to roughly zero on ~10% of frames while the
// detector is still reporting 0.8. Per-frame thresholding would call that a stranger.
Collections.sort(identity);
System.out.printf("a face in %d frames -- identity median %.3f, p10 %.3f, worst %.3f%n",
	identity.size(), identity.get(identity.size() / 2),
	identity.get(identity.size() / 10), identity.get(0));
```

video4j hands out an OpenCV `Mat` and this module keeps OpenCV off its compile path — the
dependency is **test scope**, and `FaceImage` is a plain byte array precisely so an application
holding its own OpenCV is not forced onto video4j's (currently 4.10, via `opencv-ffm`). The whole
bridge:

```java
/** OpenCV {@code CV_8UC3} is BGR, row-major, stride {@code width * 3} — FaceImage's layout. */
private static FaceImage toFaceImage(Mat mat) {
	byte[] bgr = new byte[mat.rows() * mat.cols() * 3];
	int copied = mat.get(0, 0, bgr);
	// A non-continuous or non-8UC3 Mat copies short and leaves the tail zeroed, which detects
	// as a plausible-looking nothing rather than as an error. Cheap to rule out here.
	if (copied != bgr.length) {
		throw new IllegalArgumentException("copied " + copied + " of " + bgr.length
			+ " bytes -- Mat is not continuous CV_8UC3");
	}
	return FaceImage.ofBgrBytes(mat.cols(), mat.rows(), bgr);
}
```

Three things that example is doing on purpose:

- **Frames are closed.** video4j allocates a `Mat` per frame and native memory is invisible to the
  garbage collector, so a long clip without the try-with-resources grows until the OS intervenes
  rather than the JVM.
- **No resize step.** The detector caps the long edge itself, so adding one only costs a second
  interpolation and moves the boxes.
- **Boxes are drawn after the conversion**, onto the `Mat` the viewer shows. Draw first and the
  rectangles are part of the pixels the model sees.

The types used above come from `facedetect4j-api` and are documented [there](../api).

## GPU is enforced, not preferred

`Yunet4j.pipeline(dir)` requests CUDA and **throws** if the provider will not attach. It never falls
back to CPU on its own — a silent fallback turns a deployment error into a performance mystery where
nothing fails, nothing logs, and the only symptom is throughput an order of magnitude low. Opt out
explicitly:

```java
Yunet4j.pipeline(dir, Device.cpu());
Yunet4j.pipeline(dir, Device.cuda(1));   // second GPU
```

### Making CUDA load

ONNX Runtime 1.28 ships **two different native builds of the same version**, and Maven gets the
older one:

```
$ readelf -d libonnxruntime_providers_cuda.so | grep -o 'libcud.*so\.[0-9]*'
  Maven  onnxruntime_gpu:1.28.0 : libcudart.so.12  libcublas.so.12  libcublasLt.so.12
  PyPI   onnxruntime-gpu 1.28.0 : libcudart.so.13  libcublas.so.13  libcublasLt.so.13
```

onnxruntime.ai's "1.27+ is CUDA 13" table describes the PyPI and NuGet packages, not the Java one.
So on a CUDA 13 host the Maven jar cannot load its provider, and the error names no library at all:

```
OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library
```

Worse if some *other* package left an old CUDA 12 behind — Debian ships 12.4 as `libcudart12`,
which many things pull in. Then the provider gets far enough to bind symbols and fails naming a
function nobody has heard of, with no hint that the version is the problem:

```
libonnxruntime_providers_cuda.so: undefined symbol: cudaLibraryGetKernel, version libcudart.so.12
```

**You do not need to install CUDA 12 for this.** `setup-cuda.sh` detects what the machine has:

| System has | What happens | Downloaded |
|---|---|---|
| CUDA 13 | uses it, with ORT's CUDA 13 natives from the PyPI wheel | ORT natives + cuDNN, ~1.1 GB |
| CUDA 12 | uses it as-is with the stock Maven build | cuDNN only, if missing |
| neither | fetches a project-local CUDA 12 | ~2.3 GB |

```bash
./setup-cuda.sh
mvn test
```

**Nothing to source, no variables to export.** The script only puts files under `.cuda/`;
`CudaNatives` finds them from inside the JVM, before any ONNX Runtime class initialises. That
matters because otherwise every launcher has to be taught the same environment — shell, IDE run
configuration, surefire fork, and any downstream application. `FORCE_CUDA12=1` opts back into the
stock Maven build.

Two things have to happen, and neither needs the loader's search path:

- **`onnxruntime.native.path`** is a system property, so it can simply be set — as long as it is set
  first. ONNX Runtime reads it once, in its own class initialiser.
- **cuDNN** is `dlopen`'d by name from a directory the loader does not search. `System.load` on the
  absolute path registers the library under its `SONAME`, so ORT's later `dlopen` finds it already
  resident and never consults the search path at all. Load order is not computed — cuDNN 9 is a
  dozen interdependent libraries — the loads are just retried to a fixpoint.

If this library is a **dependency** rather than the build you are running, the search from the
working directory upward will not reach into this checkout. Install once for the machine with
`./setup-cuda.sh --global` (`~/.facedetect4j/cuda`), or point at any directory with
`-Dfacedetect4j.cuda.dir=`.

⚠️ The CUDA 13 path is an **unsupported combination**: the core and provider libraries come from
the PyPI wheel while `libonnxruntime4j_jni.so` still comes from the Maven jar, because the wheel
has no Java shim. It works because both are ORT 1.28.0 and the wheel's core carries the SONAME
(`libonnxruntime.so.1`) the shim links against — so **the two versions must be kept equal**. Bump
`ORT_VERSION` in the script together with `ort.version` in the parent pom.

#### With root

CUDA itself is better installed from NVIDIA's packages — the debs drop an `/etc/ld.so.conf.d`
entry, so after `ldconfig` those libraries are on the default loader path and `setup-cuda.sh`
detects and uses them instead of downloading a second copy. For CUDA **12** you need the
**`debian12`** repo; `debian13` starts at CUDA 13.1.

cuDNN cannot be installed this way on Debian: `compute/cudnn/repos` has no Debian tree (both
`debian12` and `debian13` are 404), and the `debian13` CUDA repo carries zero cuDNN packages. The
tarball this script fetches is the only route.

#### cuDNN is required and invisible

It does not appear in the `NEEDED` list above, because ORT `dlopen`s `libcudnn.so` lazily at the
first Conv node. Absence from `NEEDED` means "not linked at load time", not "not required" — the
session opens fine and inference then fails with:

```
ORT_NOT_IMPLEMENTED ... cuDNN is unavailable or disabled for CUDA Execution Provider
```

Without a working GPU the tests **skip** rather than fail, so a green build is not evidence the GPU
path works — check for `Skipped: 0`.

## Models

Not bundled — fetch them (note the `media.` host; opencv_zoo is git-lfs, and
`raw.githubusercontent.com` serves a ~130-byte text pointer named `.onnx` that fails much later with
an opaque protobuf error):

```bash
mkdir -p models
B=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
curl -L -o models/face_detection_yunet_2026may.onnx    $B/face_detection_yunet/face_detection_yunet_2026may.onnx
curl -L -o models/face_recognition_sface_2021dec.onnx  $B/face_recognition_sface/face_recognition_sface_2021dec.onnx
```

`2026may` has a dynamic input shape and is preferred. `2023mar` is fixed at 640x640 and is
letterboxed; whichever is present is detected from the graph, not configured.

## The things that will cost you a day

**Input must be padded to a multiple of 32.** Mandatory, not cosmetic. YuNet's neck adds an
upsampled stride-32 map to the stride-16 one, and at ragged sizes those paths round differently so
the addition cannot broadcast. Measured on `2026may`:

```
704x704  OK          677x687  FAIL: Add_44 ... broadcast an axis by a dimension
704x992  OK                         other than 1. 42 by 43
```

OpenCV's DNN engine tolerates ragged sizes; ONNX Runtime does not. A port that forwards
`setInputSize(w, h)` therefore works on some images and throws on others, decided by nothing but
image dimensions. Cost of the padding: 0.06 px box drift on 677x687, 4.4 px on 675x963, versus
OpenCV.

**Both models take raw BGR 0-255.** No mean subtraction, no scaling, no channel swap — OpenCV drives
both with a bare `blobFromImage`. The ArcFace-family `(RGB-127.5)/127.5` and SCRFD's `/128` are both
wrong here and produce embeddings that still cluster, still compare, and are quietly much worse.

**YuNet's score is `sqrt(cls * obj)`**, not either head alone. Box centres are plain cell offsets but
width and height are `exp()` — applying `exp` to all four gives plausibly-placed, wrongly-sized boxes.

**Create the ORT environment before touching `SessionOptions`.** `addCUDA()` logs during provider
registration, so calling it first fails with `Attempt to use DefaultLogger but none has been
registered` — which reads like a missing-CUDA error and is not one.

**SFace declares 175 graph inputs.** The mxnet export left every weight exposed as an input; all but
`data` carry initializers and are optional. Iterating `getInputNames()` and feeding them all fails
confusingly.

**YuNet keypoints are already in ArcFace order.** OpenCV documents slots 4-5 as "right eye", but that
is anatomical (subject's right = image left), and ArcFace slot 0 is simply the image-left point. Both
land on the same array order. Permuting them "to fix the naming" yields a mirrored crop that still
embeds cleanly and merely scores worse.

**Do not threshold identity per frame in video.** Measured over the 887 frames of the rotation clip
in the example, against a frontal reference:

```
median +0.892     p25 +0.638     p10 +0.260     worst -0.092
```

The worst frames are not detection failures — the detector still reports 0.63-0.84 there. At full
profile SFace sees one eye and no mouth corners, so the embedding is close to orthogonal to a
frontal one, and roughly one frame in ten of a turning head falls below the 0.3 band that
distinguishes *different people*. A per-frame threshold calls that a stranger. Aggregate first:
take the best-of-N over a window, or gate on pose before embedding at all.

## Validation

Alignment and decoding are checked against reference implementations, not against a second copy of
the same maths:

| What | Against | Result |
|---|---|---|
| Umeyama similarity transform | `skimage.transform.SimilarityTransform` | 1e-9 |
| Full alignment + embedding | `cv2 FaceRecognizerSF.alignCrop` | cosine >= 0.99994 |
| YuNet decode (box, score) | `cv2 FaceDetectorYN`, same weights | <= 0.07 px on aligned sizes |
| WIDER FACE AP | OpenCV's published figures | within 0.003 |

`mvn test` runs these; tests skip rather than fail when models or a GPU are absent.

## Status and gaps

* **Threshold defaults** are YuNet's usual band (score 0.6, NMS 0.3). SFace's useful similarity
  cut-off is around 0.30-0.36 — calibrate against your own data, and do not carry over
  InspireFace's 0.48.
* **No batching.** One image per call. Batched inference would help throughput on video and is the
  obvious next addition.
* **No face tracking** across frames.
* **CPU path exists but is untested for performance** — it is an escape hatch, not a supported mode.

## Attribution

Portions of the code in this project were co-authored with the assistance of AI.
