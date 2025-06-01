cmake_minimum_required(VERSION 3.0.0)
project(yolo_ort)

option(INSPIREFACE_INCLUDE_DIR "Path to source InspireFace include directory." STRING)
message(STATUS "INSPIREFACE_INCLUDE_DIR: ${INSPIREFACE_INCLUDE_DIR}")

option(INSPIREFACE_LIB_DIR "Path to built InspireFace directory." STRING)
message(STATUS "INSPIREFACE_LIB_DIR: ${INSPIREFACE_LIB_DIR}")

option(TENSOR_RT_DIR "Path to built TensorRT directory." STRING)
message(STATUS "TENSOR_RT_DIR: ${TENSOR_RT_DIR}")

find_package(OpenCV REQUIRED)

#include_directories("../InspireFace/cpp/inspireface/c_api/")
#include_directories("../InspireFace/cpp/inspireface/include/inspireface/")
#include_directories("../InspireFace/cpp/")
#include_directories("../InspireFace/3rdparty/InspireCV/include/")
#include_directories("../InspireFace/3rdparty/InspireCV/src/")
#include_directories("../InspireFace/cpp/inspireface")
#include_directories("../InspireFace/3rdparty/yaml-cpp/include")
#include_directories("../InspireFace/3rdparty/MNN/include")


include_directories("${INSPIREFACE_INCLUDE_DIR}")


add_library(jinspireface SHARED
    src/jinspireface.cpp)

set_target_properties(jinspireface PROPERTIES VERSION "1.0.0")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#target_include_directories(jinspireface PUBLIC "${INSPIREFACE_SRC_DIR}/include")
#target_link_libraries(jinspireface "${TENSOR_RT_DIR}/lib")
#target_link_libraries(jinspireface "${TENSOR_RT_DIR}/targets/x86_64-linux-gnu/lib")

target_compile_features(jinspireface PRIVATE cxx_std_14)
target_link_libraries(jinspireface ${OpenCV_LIBS})
target_link_libraries(jinspireface "${INSPIREFACE_LIB_DIR}/libInspireFace.so")


#add_executable(BoxSample src/box_sample.cpp)
##target_link_libraries(BoxSample InspireFace ${ext})
#target_link_libraries(BoxSample "${TENSOR_RT_DIR}/lib")
#target_link_libraries(BoxSample "${TENSOR_RT_DIR}/targets/x86_64-linux-gnu/lib")
#target_link_libraries(BoxSample "${INSPIREFACE_LIB_DIR}/libInspireFace.so")
#target_link_libraries(BoxSample ${OpenCV_LIBS})
#set_target_properties(BoxSample PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/sample/"
#)


if(APPLE)
set_target_properties(jinspireface PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/macosx)
endif()

if(UNIX AND NOT APPLE)
set_target_properties(jinspireface PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/linux)
endif()