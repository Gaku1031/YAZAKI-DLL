cmake_minimum_required(VERSION 3.15)
project(CppBloodPressureDLL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
endif()

# Find packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Try to find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/include
        ENV ONNXRUNTIME_ROOT_PATH
    PATH_SUFFIXES include
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime/lib
        ENV ONNXRUNTIME_ROOT_PATH
    PATH_SUFFIXES lib
)

# Find MediaPipe (required)
find_path(MEDIAPIPE_INCLUDE_DIR
    NAMES mediapipe/framework/calculator_framework.h
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediapipe
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mediapipe
        ENV MEDIAPIPE_ROOT_PATH
    PATH_SUFFIXES include
    REQUIRED
)

find_library(MEDIAPIPE_LIB
    NAMES mediapipe_framework
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediapipe/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mediapipe/lib
        ENV MEDIAPIPE_ROOT_PATH
    PATH_SUFFIXES lib
    REQUIRED
)

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(ONNXRUNTIME_INCLUDE_DIR)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    message(WARNING "ONNX Runtime not found. Please install or set ONNXRUNTIME_ROOT_PATH")
endif()

# MediaPipe is required
include_directories(${MEDIAPIPE_INCLUDE_DIR})
message(STATUS "Found MediaPipe include: ${MEDIAPIPE_INCLUDE_DIR}")
add_definitions(-DMEDIAPIPE_AVAILABLE)

# Source files
file(GLOB SRC_FILES src/*.cpp)

# Create shared library
add_library(BloodPressureDLL SHARED ${SRC_FILES})

# Link libraries
target_link_libraries(BloodPressureDLL
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

if(ONNXRUNTIME_LIB)
    target_link_libraries(BloodPressureDLL ${ONNXRUNTIME_LIB})
    message(STATUS "Linked ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    message(WARNING "ONNX Runtime library not found")
endif()

# MediaPipe is required
target_link_libraries(BloodPressureDLL ${MEDIAPIPE_LIB})
message(STATUS "Linked MediaPipe: ${MEDIAPIPE_LIB}")

# Windows-specific settings
if(WIN32)
    set_target_properties(BloodPressureDLL PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
    )
endif()

# Set output directory
set_target_properties(BloodPressureDLL PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
) 
