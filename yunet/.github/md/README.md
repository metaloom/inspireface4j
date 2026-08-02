# Facedetect4J YuNet

GPU face **detection** (YuNet) and **embedding** (SFace) for Java, on ONNX Runtime.

The permissively licensed backend of [Facedetect4J](../): both models carry explicit commercial
grants — YuNet is **MIT**, SFace is **Apache-2.0** — where the InspireFace and InsightFace model
packs are stated by their authors to be for academic use only.

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-yunet</artifactId>
  <version>${project.version}</version>
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
%{snippet|id=detect-usage.example|file=src/test/java/io/metaloom/facedetect4j/yunet/example/UsageExampleTest.java}
```

Embedding:

```java
%{snippet|id=embed-usage.example|file=src/test/java/io/metaloom/facedetect4j/yunet/example/UsageExampleTest.java}
```

Feeding pixels from an existing decoder skips AWT entirely:

```java
FaceImage img = FaceImage.ofBgrBytes(width, height, bgrBytes);
```

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

The Maven `onnxruntime_gpu` artifact is a **CUDA 12** build, even though onnxruntime.ai documents
1.27+ as CUDA 13 — that table describes the PyPI and NuGet packages, not the Java one. Verified:

```
$ readelf -d libonnxruntime_providers_cuda.so | grep NEEDED
  libcudart.so.12   libcublas.so.12   libcublasLt.so.12   libcurand.so.10
```

No `libcudnn` soname appears anywhere, so **cuDNN is not required**. On a host with only CUDA 13
(Debian trixie), supply CUDA 12 from pip wheels and export the path **before the JVM starts** — the
dynamic loader fixes its search path at process start, so setting it from inside Java is too late:

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12
export LD_LIBRARY_PATH="$(ls -d .venv-cuda/lib/python*/site-packages/nvidia/*/lib | paste -sd:)"
```

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
