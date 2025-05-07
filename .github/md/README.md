# InspireFace4J


## Limitations

Currently only AMD64 Linux is supported. Support for other platforms is not planned.

## Usage

```xml
<dependency>
  <groupId>io.metaloom.inspireface4j</groupId>
  <artifactId>inspireface4j</artifactId>
  <version>${project.version}</version>
</dependency>
```

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

- [InspireFace](https://github.com/HyperInspire/InspireFace)
- JDK 23 or newer
- Maven
- GCC 13

### Building native code

```bash
#git clone git@github.com:HyperInspire/InspireFace.git
git clone git@github.com:deepinsight/insightface.git  (Head Rev: efb5639ec66d4e94004e4d16f34f44630179f95a)

# Fix for minor path issue:
# cd insightface/cpp-package/inspireface/cpp/inspireface
# mv Initialization_module/ initialization_module/

cd jinspirelib
./build.sh
```

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```

