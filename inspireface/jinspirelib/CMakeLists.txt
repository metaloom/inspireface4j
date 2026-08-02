cmake_minimum_required(VERSION 3.16)
project(jinspireface)

option(INSPIREFACE_INCLUDE_DIR "Path to source InspireFace include directory." STRING)
message(STATUS "INSPIREFACE_INCLUDE_DIR: ${INSPIREFACE_INCLUDE_DIR}")

option(INSPIREFACE_LIB_DIR "Path to built InspireFace directory." STRING)
message(STATUS "INSPIREFACE_LIB_DIR: ${INSPIREFACE_LIB_DIR}")

option(TENSOR_RT_DIR "Path to built TensorRT directory." STRING)
message(STATUS "TENSOR_RT_DIR: ${TENSOR_RT_DIR}")

# Set OpenCV_DIR via cmake -DOpenCV_DIR=... or the environment, so the OpenCV linked here is
# chosen rather than whichever one happens to be installed system wide.
if(DEFINED ENV{OpenCV_DIR} AND NOT OpenCV_DIR)
    set(OpenCV_DIR $ENV{OpenCV_DIR} CACHE PATH "Path to the OpenCV build directory")
endif()

if(OpenCV_DIR)
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
else()
    find_package(OpenCV REQUIRED)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# This must be the same OpenCV major that opencv-ffm binds, which the Java side reaches through
# video4j (io.metaloom.opencv:opencv-ffm, versioned to track its OpenCV: 4.10.0 -> OpenCV 4.10).
# video4j allocates the cv::Mat and this library dereferences the pointer, so a mismatch is not a
# link error and not a missing symbol - it is one OpenCV's struct read through another's headers,
# in the same address space, with both libraries loaded. Wrong-but-plausible pixel data at best.
set(REQUIRED_OPENCV_MAJOR 4 CACHE STRING "OpenCV major version that opencv-ffm is built against")
if(NOT OpenCV_VERSION_MAJOR VERSION_EQUAL REQUIRED_OPENCV_MAJOR)
    message(FATAL_ERROR
        "OpenCV ${REQUIRED_OPENCV_MAJOR}.x is required to match opencv-ffm, but found "
        "${OpenCV_VERSION}. Pass -DOpenCV_DIR=<build-dir>, and if opencv-ffm itself has moved "
        "major version, -DREQUIRED_OPENCV_MAJOR=<n> together with it.")
endif()

#include_directories("../InspireFace/cpp/inspireface/c_api/")
#include_directories("../InspireFace/cpp/inspireface/include/inspireface/")
#include_directories("../InspireFace/cpp/")
#include_directories("../InspireFace/3rdparty/InspireCV/include/")
#include_directories("../InspireFace/3rdparty/InspireCV/src/")
#include_directories("../InspireFace/cpp/inspireface")
#include_directories("../InspireFace/3rdparty/yaml-cpp/include")
#include_directories("../InspireFace/3rdparty/MNN/include")

include_directories("${INSPIREFACE_INCLUDE_DIR}")

#target_include_directories(jinspireface PUBLIC "${INSPIREFACE_SRC_DIR}/include")
#target_link_libraries(jinspireface "${TENSOR_RT_DIR}/lib")
#target_link_libraries(jinspireface "${TENSOR_RT_DIR}/targets/x86_64-linux-gnu/lib")

add_library(jinspireface SHARED
    src/jinspireface.cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


target_compile_features(jinspireface PRIVATE cxx_std_17)
target_link_libraries(jinspireface ${OpenCV_LIBS})
target_link_libraries(jinspireface "${INSPIREFACE_LIB_DIR}/libInspireFace.so")


add_executable(BoxSample src/box_sample.cpp)
#target_link_libraries(BoxSample InspireFace ${ext})
#target_link_libraries(BoxSample "${TENSOR_RT_DIR}/lib")
#target_link_libraries(BoxSample "${TENSOR_RT_DIR}/targets/x86_64-linux-gnu/lib")
target_link_libraries(BoxSample "${INSPIREFACE_LIB_DIR}/libInspireFace.so")
target_link_libraries(BoxSample ${OpenCV_LIBS})
set_target_properties(BoxSample PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
)


if(APPLE)
set_target_properties(jinspireface PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/macosx)
endif()

if(UNIX AND NOT APPLE)
set_target_properties(jinspireface PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/linux)
endif()