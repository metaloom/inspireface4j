# InspireFace4J


## Limitations

Currently only AMD64 Linux is supported. Support for other platforms is not planned.

## Usage

```xml
<dependency>
  <groupId>io.metaloom.inspireface4j</groupId>
  <artifactId>inspireface4j</artifactId>
  <version>0.0.1-SNAPSHOT</version>
</dependency>
```

## Models

```
mkdir packs && cd packs
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Pikachu
wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron

# Not supported:
# wget https://github.com/HyperInspire/InspireFace/releases/download/v1.x/Megatron_TRT

```

## Examples

Image Example
```java
String imagePath = "src/test/resources/pexels-olly-3812743_1k_lower.jpg";

// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
Video4j.init();

InspirefaceLib.init(DEFAULT_PACK, 640);

// Load the image and invoke the detection
BufferedImage img = ImageUtils.load(new File(imagePath));
Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
CVUtils.bufferedImageToMat(img, imageMat);

// Invoke the detection
FaceDetections detections = InspirefaceLib.detect(imageMat, true);

// Print the detections
for (Detection detection : detections) {
	System.out.println(detection.box() + " @ " + detection.conf());
}
// Show the result and release the mat and the detection data
ImageUtils.show(imageMat);
MatProvider.released(imageMat);

```


Video Example
```java

// Initialize video4j and InspirefaceLib (Video4j is used to handle OpenCV Mat)
Video4j.init();
InspirefaceLib.init("packs/Pikachu", 640);
SimpleImageViewer viewer = new SimpleImageViewer();

// Open the video using Video4j
try (VideoFile video = VideoFile.open("src/test/resources/8090198-hd_1366_720_25fps.mp4")) {

	// Process each frame
	VideoFrame frame;
	while ((frame = video.frame()) != null) {
		// System.out.println(frame);

		// Optionally downscale the frame
		CVUtils.resize(frame, 512);

		// Run the detection on the mat reference
		FaceDetections detections = InspirefaceLib.detect(frame.mat(), true);

		if (!detections.isEmpty()) {
			// Extract the face embedding from the first face
			float[] embedding = InspirefaceLib.embedding(frame.mat(), detections, 0);
			// Extract the face attributes
			List<FaceAttributes> attrs = InspirefaceLib.attributes(frame.mat(), detections, true);
		}

		// Print the detections
		for (Detection detection : detections) {
			double confidence = detection.conf();
			BoundingBox box = detection.box();
			System.out.println("Frame[" + video.currentFrame() + "] = " + confidence + " @ " + box);
		}

		viewer.show(frame.mat());
	}
}
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

The CMakeLists.txt needs to be adapted to include all the different sources (e.g inspireface + inspirecv)

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

