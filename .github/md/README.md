# Facedetect4J

Facedetect4J provides face **detection** (bounding box, landmarks) and face **embedding extraction**
for Java, behind one backend neutral API.

Three backends ship today:

| Module | Backend | Runs on | Model licence |
|---|---|---|---|
| [`yunet`](yunet) | YuNet + SFace via ONNX Runtime | **GPU** (CUDA) | 🟡 MIT + Apache-2.0 — commercial use granted |
| [`inspireface`](inspireface) | [InspireFace](https://github.com/HyperInspire/InspireFace) via FFM | CPU (MNN) | 🔴 non-commercial only |
| [`jdlib`](jdlib) | [dlib](http://dlib.net) via JNI | CPU, or GPU for the CNN detector | 🟡 depends on which models you load |

The licence column is the reason this project exists. InspireFace and InsightFace publish excellent
models whose weights their authors state are for academic use only, which makes them unusable in a
shipped product regardless of how well they score. The `yunet` backend is measurably close and
carries explicit commercial grants — see [its README](yunet) for the numbers.

`jdlib` is the trap worth knowing about: dlib's own code is Boost licensed and permissive, but the
68 point shape predictor everyone reaches for first is **not** licensed for commercial use, and
every jdlib entry point except plain `detectFace` needs a shape predictor. The 5 point predictor is
a drop-in replacement with no such restriction. Nothing in the API tells you which one you loaded —
see [the jdlib README](jdlib).

## Modules

* **`api`** — `io.metaloom.facedetect4j.api`. Names no model, no inference runtime and no native
  library, and depends on nothing but the JDK. This is what consuming code compiles against. Its
  test-jar carries `AbstractFacePipelineTest`, the conformance suite every backend is run against,
  and `TestData`, the locator for the shared media in [`testdata/`](testdata).
* **`yunet`** — GPU detection and embedding on ONNX Runtime. Permissively licensed models.
* **`inspireface`** — FFM binding to InspireFace. Also exposes attributes, 106 landmarks and Euler
  angles, which the neutral API does not model.
* **`jdlib`** — JNI binding to dlib. HOG and CNN (MMOD) detection, 5/68 point landmarks and 128-d
  embeddings.

## One API, three backends

All three implement `FacePipeline`, and all three are run against the same conformance suite from
the `api` test-jar — which is what makes them interchangeable rather than merely similar. Where a
backend cannot do something it **refuses**, and the suite asserts the refusal:

| | `yunet` | `inspireface` | `jdlib` |
|---|---|---|---|
| `detect` | ✅ | ✅ | ✅ (no confidence — every score is 1.0) |
| ArcFace landmarks | ✅ | ❌ 106-point mapping undocumented | ✅ only with the 68-point predictor |
| `embed(image, face)` | ✅ | ✅ | ✅ |
| `embed(AlignedFace)` | ✅ | ❌ no crop entry point | ❌ no crop entry point |
| `align` | ✅ | ❌ | ✅ only with the 68-point predictor |
| Device | **cuda** | cpu (MNN) | cpu |

`supportsAlignedEmbed()` answers the ❌ rows without catching an exception. It matters more than it
looks: a backend that can only embed from its own alignment cannot be compared like-for-like with
one that accepts a shared crop, so a single ranking across both silently folds alignment quality
into what reads as an embedding-quality result.

## Usage

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-yunet</artifactId>
  <version>${project.version}</version>
</dependency>
```

The API module comes in transitively; depend on `facedetect4j-api` directly only where a module must
compile against the interfaces without pulling in a backend.

```java
try (FacePipeline faces = Yunet4j.pipeline(Path.of("models"))) {
    FaceImage img = FaceImage.read(Path.of("photo.jpg"));

    for (Face face : faces.detectAndEmbed(img)) {
        System.out.println(face.box() + " -> " + face.embedding().length + "d");
    }
}
```

Video processing can be done by using the library in combination with
[Video4j](https://github.com/metaloom/video4j).

## Building

```bash
# io.metaloom:maven-parent must be installed first
mvn -f ../maven-parent/pom.xml -N install

mvn clean install
```

`inspireface` and `jdlib` additionally need their native libraries built — see the module READMEs.
The prebuilt `.so` files are committed, so a plain build works; only changes to the JNI/FFM glue
require the native toolchain. Building only the permissive stack:

```bash
mvn clean install -pl api,yunet
```

## testdata/

The media the tests run on lives once, at the reactor root, rather than once per module. The two
videos alone are 43 MB, several backends want the same faces, and — the real reason — comparing
backends is only meaningful on identical input, which per-module copies cannot guarantee for long.

Reach it through `TestData` from the `api` test-jar; do not hardcode `../testdata`.

## Documentation

The `README.md` files in this repository are **generated** during the `clean` phase from
`.github/md/README.md` in each module. Edit the template, not the generated file, and let the
snippet filter inline the examples from the tests so they cannot drift from code that compiles.

## License

The code of this project is licensed under the Apache License 2.0. **The licences of the models are
separate and are not all permissive** — check the module README before shipping anything.

## Attribution

Portions of the code in this project were co-authored with the assistance of AI.
