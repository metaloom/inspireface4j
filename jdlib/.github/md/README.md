# Facedetect4J Jdlib

The dlib backend of [Facedetect4J](../): a Java wrapper for dlib on Linux and MacOSX. The wrapper
contains stubs for the HOG face detector, the CNN (MMOD) face detector, facial landmarks and face
embeddings.

> ⚠️ **Read [Model licences](#model-licences) before shipping.** dlib itself is Boost licensed and
> permissive, but the widely used 68 point shape predictor is **not licensed for commercial use** —
> and every entry point of this wrapper except `detectFace` needs a shape predictor.

# Fork Note

This is a fork of [jdlib](https://github.com/tahaemara/jdlib) which was initially created by Taha Emara.

This fork supports a way to run the CNN based face detection which can utilize the GPU.

## Demo
### Face Clustering Example

<img src="https://media.giphy.com/media/FD9AfNUw3VX8CqYaph/giphy.gif" width="700" height="400" />

## Using Jdlib

```xml
<dependency>
  <groupId>io.metaloom.facedetect4j</groupId>
  <artifactId>facedetect4j-jdlib</artifactId>
  <version>${project.version}</version>
</dependency>
```

The Java package is still `io.metaloom.jdlib`; only the Maven coordinates moved. The previously
released `io.metaloom.jdlib:jdlib:2.1.0` remains on Maven Central and still resolves — but do not
put both on one classpath, or `io.metaloom.jdlib.Jdlib` is present twice and two copies of
`libjdlib.so` get extracted and loaded.

## Model licences

The models are downloaded separately from dlib and are **not** covered by dlib's own Boost licence.
They do not all carry the same terms:

| Model | Used for | Commercial use |
|---|---|---|
| `shape_predictor_68_face_landmarks.dat` | landmarks, and the alignment step of embeddings | 🔴 **No.** Trained on iBUG 300-W, whose licence excludes commercial use; dlib states the trained model therefore cannot be used in a commercial product |
| `shape_predictor_5_face_landmarks.dat` | same, 5 points instead of 68 | 🟢 Yes. Trained on a dataset created by dlib's author and released CC0 |
| `dlib_face_recognition_resnet_model_v1.dat` | 128-d embeddings | 🟡 No non-commercial restriction stated by dlib, unlike the 68 point predictor. Verify against the current upstream wording before relying on it |
| `mmod_human_face_detector.dat` | CNN face detection | 🟡 Same as above |

For anything shipped, use the **5 point** predictor. It is a drop-in for the constructor argument
and is what the embedding path actually needs — the extra 63 points of the 68 point model are only
useful if you are drawing them. Swapping it in is the difference between a permissive stack and one
that cannot be released, and nothing in the API will warn you which one you loaded.

The [yunet](../yunet) module avoids the question entirely (MIT + Apache-2.0) and runs on the GPU.

## Runtime requirement: libmkl_rt.so

The bundled `libjdlib.so` links against Intel MKL, which is not part of any base install:

```
$ ldd src/main/resources/native/linux/libjdlib.so | grep mkl
	libmkl_rt.so => not found
```

Without it, `new Jdlib(...)` throws `UnsatisfiedLinkError: libmkl_rt.so: cannot open shared object
file`. Note this comes out of the **constructor** — `Jdlib`'s loader catches `Exception`, which does
not cover `UnsatisfiedLinkError`. (A missing `.so` *resource*, by contrast, is swallowed and only
surfaces on the first native call.)

The path must be set before the JVM starts; the dynamic loader fixes its search path at process
start, so setting it from inside Java is too late:

```bash
python3 -m venv .venv && .venv/bin/pip install mkl
ln -s libmkl_rt.so.3 .venv/lib/libmkl_rt.so     # the wheel ships only the versioned soname
export LD_LIBRARY_PATH="$PWD/.venv/lib:$LD_LIBRARY_PATH"
```

`JdlibSmokeTest` skips with this message rather than failing when MKL is absent, so the module still
builds on a machine without it — a green build is therefore not evidence that the native path works.
Check for `Skipped: 1`.

## Compiling Jdlib

### Requirements:

- Dlib installation requirements [Using dlib from C++](http://dlib.net/compile.html)
- JDK 17 or newer
- Maven
- GCC 13

```
Debian 11:

* nvidia-cuda-toolkit
* nvidia-cudnn 

```

### Compile JNI/C++ code:

```bash
cd jdlib

# Generate needed JNI header files via mvn. This is a required step of the native build, not just
# a Java build: -h jni writes io_metaloom_jdlib_Jdlib.h, which jni/CMakeLists.txt compiles against.
mvn clean compile

# Now build JNI lib
mkdir -p build && cd build

# Ensure to select a compatible compiler (Example for Debian 11)
export JAVA_HOME=/opt/jvm/java17
CC=gcc-13 CXX=/usr/bin/g++-13 cmake ../jni
make 
```

### Compile Java Package:

```bash
mvn clean package
```

After that you will have the JAR file including the binaries for your platform inside Jdlib/target. Then you can use it inside your project as an external jar or install it manually in [local maven](https://maven.apache.org/guides/mini/guide-3rd-party-jars-local.html). 

## Compiling and running examples

- Download needed models to example folder

```bash
cd examples

# The 5 point predictor is the one to use in anything shipped - see Model licences above.
# The examples draw all 68 points, which is the one case where the 68 point model earns its licence.
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
wget http://dlib.net/files/mmod_human_face_detector.dat.bz2

bzip2 -dk shape_predictor_5_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -dk mmod_human_face_detector.dat.bz2
```

- Build and run examples

```bash
cd examples
mvn clean package
java -jar target/clustering-example-jar-with-dependencies.jar
java -jar target/landmarks-example-jar-with-dependencies.jar
java -jar target/cnn-facedetect-example-jar-with-dependencies.jar
```

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```