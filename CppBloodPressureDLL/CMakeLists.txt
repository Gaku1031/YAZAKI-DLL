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

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)
find_package(Eigen3 REQUIRED NO_MODULE)

# Try to find ONNX Runtime
if(DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    
    if(EXISTS ${ONNXRUNTIME_INCLUDE_DIR} AND EXISTS ${ONNXRUNTIME_LIB})
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_ROOT}")
        set(ONNXRUNTIME_FOUND TRUE)
    else()
        message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
        set(ONNXRUNTIME_FOUND FALSE)
    endif()
else()
    # Fallback search
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
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        set(ONNXRUNTIME_FOUND TRUE)
    else()
        set(ONNXRUNTIME_FOUND FALSE)
    endif()
endif()

# Find MediaPipe (required)
if(DEFINED MEDIAPIPE_ROOT)
    set(MEDIAPIPE_INCLUDE_DIR "${MEDIAPIPE_ROOT}/include")
    set(MEDIAPIPE_LIB "${MEDIAPIPE_ROOT}/lib/mediapipe_framework.lib")
    
    if(EXISTS ${MEDIAPIPE_INCLUDE_DIR})
        message(STATUS "Found MediaPipe: ${MEDIAPIPE_ROOT}")
        set(MEDIAPIPE_FOUND TRUE)
    else()
        message(WARNING "MediaPipe not found at ${MEDIAPIPE_ROOT}")
        set(MEDIAPIPE_FOUND FALSE)
    endif()
else()
    # Fallback search
    find_path(MEDIAPIPE_INCLUDE_DIR
        NAMES mediapipe/framework/calculator_framework.h
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediapipe
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mediapipe
            ENV MEDIAPIPE_ROOT_PATH
        PATH_SUFFIXES include
    )

    find_library(MEDIAPIPE_LIB
        NAMES mediapipe_framework face_landmark_front_cpu
        PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediapipe/lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/mediapipe/lib
            ENV MEDIAPIPE_ROOT_PATH
        PATH_SUFFIXES lib
    )
    
    if(MEDIAPIPE_INCLUDE_DIR AND MEDIAPIPE_LIB)
        set(MEDIAPIPE_FOUND TRUE)
    else()
        set(MEDIAPIPE_FOUND FALSE)
    endif()
endif()

if(NOT MEDIAPIPE_FOUND)
    message(FATAL_ERROR "MediaPipe is required but not found. Please install MediaPipe or set MEDIAPIPE_ROOT")
endif()

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(ONNXRUNTIME_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "Found ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
    add_definitions(-DONNXRUNTIME_AVAILABLE)
else()
    message(WARNING "ONNX Runtime not found. Please install or set ONNXRUNTIME_ROOT")
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

if(ONNXRUNTIME_FOUND)
    target_link_libraries(BloodPressureDLL ${ONNXRUNTIME_LIB})
    message(STATUS "Linked ONNX Runtime: ${ONNXRUNTIME_LIB}")
else()
    message(WARNING "ONNX Runtime library not found")
endif()

# MediaPipe is required
target_link_libraries(BloodPressureDLL ${MEDIAPIPE_LIB})
message(STATUS "Linked MediaPipe: ${MEDIAPIPE_LIB}")

# Additional MediaPipe dependencies
if(WIN32)
    target_link_libraries(BloodPressureDLL ws2_32)
endif()

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
