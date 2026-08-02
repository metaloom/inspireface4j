# Facedetect4J API

The backend neutral API of [Facedetect4J](../). It names no model, no inference runtime and no
native library, and depends on nothing but the JDK — which is the point: consuming code compiles
against this module, so swapping backends becomes a change of construction site rather than a change
of call sites.

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-api</artifactId>
  <version>${project.version}</version>
</dependency>
```

## Types

| Type | Purpose |
|---|---|
| `FaceDetector` | `detect(FaceImage) -> List<Face>` |
| `FaceEmbedder` | `embed(FaceImage, Face)` and `embed(AlignedFace)` |
| `FacePipeline` | both, plus `detectAndEmbed`, `primaryFace`, `align` |
| `Face` | box, score, landmarks and (optionally) the embedding |
| `FacePose` | roll / yaw / pitch in degrees, plus `isFrontal(maxDegrees)` |
| `BoundingBox`, `Landmarks`, `AlignedFace` | value types |
| `FaceImage` | BGR pixels; `read(Path)`, or `ofBgrBytes(w, h, byte[])` to skip AWT |
| `Device` | `cpu()`, `cuda()`, `cuda(int)` |
| `align.ArcFaceAlign` | the canonical 5-point 112x112 warp |

## Four decisions worth knowing

**`detect()` returns every face**, unsorted and uncapped. Choosing a primary face is caller policy —
`FacePipeline.primaryFace` implements the usual largest-and-most-central rule — and baking that into
the detector throws away information the caller cannot recover.

**There are two embed entry points.** `embed(AlignedFace)` takes pixels the caller has already
aligned. Without it there is no way to run two embedders on identical input, and therefore no way to
tell whether one model beats another on embedding quality or merely on alignment quality.

**Alignment lives here, once.** `ArcFaceAlign` holds the canonical ArcFace 5-point template — the
same constants insightface and OpenCV's `FaceRecognizerSF::alignCrop` use, byte for byte. Every
backend that produces 5 landmarks shares it rather than reimplementing it, because a subtly
different template yields crops that still embed cleanly and merely score worse, which is close to
undetectable without a reference to compare against.

**`FacePose.isFrontal()` ignores roll**, and the asymmetry is the point. Roll is in-plane, so
alignment rotates it away for free and a rolled face embeds exactly as well as an upright one. Yaw
and pitch rotate half the face out of view, and no 2D warp brings it back — so they, and not the
detection score, are what predicts whether an embedding is worth comparing. `Face.estimatePose()`
is an `Optional` because several backends return no landmarks at all; a zero pose would read as
"perfectly frontal" and pass every gate. `Landmarks.estimatePose()` documents which of the three
angles is exact and which two are estimated, and why.

## The test-jar

`facedetect4j-api` also publishes a **test-jar**, which the backends depend on at test scope:

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-api</artifactId>
  <type>test-jar</type>
  <scope>test</scope>
</dependency>
```

It carries two things:

* **`AbstractFacePipelineTest`** — the contract every backend is run against. Checking each backend
  against its own expectations proves it works; checking all of them against one suite proves they
  are *interchangeable*, which is the property that makes swapping a backend a change of
  construction site rather than an audit of every call site. Backends that cannot do something must
  refuse, and the suite asserts the refusal as firmly as the positive case.
* **`TestData`** — locator for the shared media in `facedetect4j/testdata`. It walks up from the
  working directory rather than hardcoding `../testdata`, because Maven runs tests from the module
  while IDEs commonly use the reactor root. The media stays on disk and is never packaged.

## Attribution

Portions of the code in this project were co-authored with the assistance of AI.
