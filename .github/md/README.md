# InspireFace4J

InspireFace4J provides a Java-Native binding using FFM to the [InspireFace](https://github.com/HyperInspire/InspireFace) face detection library.

Supported features:

* Face detection (Boundingbox + Confidence)
* Face attribute extraction
* Face embedding extraction

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

- [InspireFace 1.2.1](https://github.com/HyperInspire/InspireFace)
- JDK 23 or newer
- Maven
- GCC 13

### Building native code

```bash
# Download and extract inspireface-linux-x86-ubuntu18-1.2.1.zip from https://github.com/HyperInspire/InspireFace/releases
cd  inspireface4j
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.2.1/inspireface-linux-x86-ubuntu18-1.2.1.zip
unp inspireface-linux-x86-ubuntu18-1.2.1.zip

cd jinspirelib
./build.sh
```

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

I tried to build the project using TensorRT + CUDA but failed. I also had issues with `inspireface-linux-tensorrt-cuda12.2_ubuntu22.04-1.2.1.zip`.

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```

