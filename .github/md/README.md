# InspireFace4J

InspireFace4J provides a Java-Native binding using FFM to the [InspireFace](https://github.com/HyperInspire/InspireFace) face detection library.

Supported features:

* Face detection (Boundingbox + Confidence)
* Face attribute extraction
* Face embedding extraction
* Face landmark extraction
* Face orientation angles (yaw, pitch, roll)


Video processing can be by using the libary in combination with [Video4j](https://github.com/metaloom/video4j).

![VideoPlayer](.github/md/output.gif)

## Limitations

Currently only AMD64 Linux is supported. Support for other platforms is not planned.
CUDA support using TensorRT is currently not working.

## Usage

```xml
<dependency>
  <groupId>io.metaloom.inspireface4j</groupId>
  <artifactId>inspireface4j</artifactId>
  <version>${project.version}</version>
</dependency>
```

## License

The code of this project is "Apache License" but the license of the models may be different. Please 

> The licensing of the open-source models employed by InspireFace adheres to the same requirements as InsightFace, specifying their use solely for academic purposes and explicitly prohibiting commercial applications.

## Models

```
mkdir packs && cd packs
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Pikachu
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron

# Not supported (Missing TensorRT support)
# wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron_TRT

```

## Examples

Image Example
```java
%{snippet|id=image-usage.example|file=src/test/java/io/metaloom/inspireface4j/example/UsageExampleTest.java}
```


Video Example
```java
%{snippet|id=video-usage.example|file=src/test/java/io/metaloom/inspireface4j/example/UsageExampleTest.java}
```


## Build 

### Requirements:

- [InspireFace 1.2.3](https://github.com/HyperInspire/InspireFace)
- [OpenCV 5.x](https://github.com/opencv/opencv) (the same build that [opencv-ffm](https://github.com/metaloom/opencv-ffm) is built against)
- JDK 25 or newer
- Maven
- GCC 13 or newer
- CMake 3.16+
- Python 3 (used to clear the executable stack flag of the shipped `libInspireFace.so`)

### Building native code

```bash
# Download and extract inspireface-linux-x86-ubuntu18-1.2.3.zip from https://github.com/HyperInspire/InspireFace/releases
cd  inspireface4j
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.2.3/inspireface-linux-x86-ubuntu18-1.2.3.zip
unp inspireface-linux-x86-ubuntu18-1.2.3.zip

cd jinspirelib
# The OpenCV 5 build directory (the one that contains OpenCVConfig.cmake).
# Defaults to ../../opencv/build - pass it explicitly or via the OpenCV_DIR env var.
./build.sh /path/to/opencv/build
```

The script builds `libjinspireface.so` into `src/main/resources/native/linux` and copies the
matching `libInspireFace.so` next to it.

A different InspireFace release can be selected via `INSPIREFACE_VERSION=1.2.x ./build.sh`.

### Notes on the shipped InspireFace binary

The upstream 1.2.3 Linux release is linked with an executable stack (`GNU_STACK` = `RWE`), which
the JVM refuses to load:

```
UnsatisfiedLinkError: cannot enable executable stack as shared object requires
```

`build.sh` therefore runs `jinspirelib/clear-execstack.py` on the bundled copy of the library,
which clears the flag (the equivalent of `execstack -c`).

### Notes on OpenCV 5

OpenCV 5 changed `CV_CN_SHIFT` from 3 to 5, so the numeric values of the `CV_8UC3` and friends
type constants differ from OpenCV 4. Always create Mats via `CvType` constants - a mat created
with a stale type value silently holds garbage and the drawing calls will fail with
`img.depth() == CV_8U` assertions.

### Notes for building from source

The `CMakeLists.txt` needs to be adapted to include all the different sources (e.g inspireface + inspirecv)

```bash
# Clone inspireface - my Head Rev: efb5639ec66d4e94004e4d16f34f44630179f95a
git clone git@github.com:HyperInspire/InspireFace.git
git clone git@github.com:deepinsight/insightface.git 

# Fix for minor path issue:
# cd insightface/cpp-package/inspireface/cpp/inspireface
# mv Initialization_module/ initialization_module/
```

### CUDA support

I tried to build the project using TensorRT + CUDA but failed. I also had issues with `inspireface-linux-tensorrt-cuda12.2_ubuntu22.04-1.2.x.zip`.

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```

## Attribution

Portions of the code in this project were co-authored with the assistance of AI.

