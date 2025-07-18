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
  zlib
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.3
)
FetchContent_MakeAvailable(zlib)
find_package(ZLIB REQUIRED)

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
        message(STATUS "Checking ONNXRUNTIME_ROOT contents:")
        if(EXISTS ${ONNXRUNTIME_ROOT})
            file(GLOB ONNX_CONTENTS "${ONNXRUNTIME_ROOT}/*")
            foreach(item ${ONNX_CONTENTS})
                message(STATUS "  ${item}")
            endforeach()
        endif()
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
    # ユーザーが指定したパスを使用
    set(MEDIAPIPE_BASE_DIR "${MEDIAPIPE_ROOT}")
else()
    # デフォルトパス
    set(MEDIAPIPE_BASE_DIR "${CMAKE_SOURCE_DIR}/../libmediapipe/output")
endif()

# libmediapipeの出力ディレクトリを検索
file(GLOB MEDIAPIPE_OUTPUT_DIRS LIST_DIRECTORIES true "${MEDIAPIPE_BASE_DIR}/*")
list(FILTER MEDIAPIPE_OUTPUT_DIRS INCLUDE REGEX ".*libmediapipe.*")

if(MEDIAPIPE_OUTPUT_DIRS)
    list(GET MEDIAPIPE_OUTPUT_DIRS 0 MEDIAPIPE_ROOT)
    message(STATUS "Found libmediapipe output directory: ${MEDIAPIPE_ROOT}")
    set(MEDIAPIPE_INCLUDE_DIR "${MEDIAPIPE_ROOT}/include")
    set(MEDIAPIPE_LIB "${MEDIAPIPE_ROOT}/lib/mediapipe.lib")
    
    if(EXISTS ${MEDIAPIPE_INCLUDE_DIR} AND EXISTS ${MEDIAPIPE_LIB})
        message(STATUS "Found libmediapipe: ${MEDIAPIPE_ROOT}")
        set(MEDIAPIPE_FOUND TRUE)
    else()
        message(WARNING "libmediapipe files not found at ${MEDIAPIPE_ROOT}")
        set(MEDIAPIPE_FOUND FALSE)
    endif()
else()
    message(WARNING "libmediapipe output directory not found in ${MEDIAPIPE_BASE_DIR}")
    set(MEDIAPIPE_FOUND FALSE)
endif()

# libmediapipeが見つからない場合の代替手段
if(NOT MEDIAPIPE_FOUND)
    message(STATUS "libmediapipe not found, checking for direct MediaPipe installation...")
    
    # MediaPipeの直接インストールを確認
    set(MEDIAPIPE_DIR "${CMAKE_SOURCE_DIR}/../mediapipe")
    if(EXISTS "${MEDIAPIPE_DIR}/bazel-bin")
        message(STATUS "Found MediaPipe bazel-bin directory: ${MEDIAPIPE_DIR}/bazel-bin")
        set(MEDIAPIPE_INCLUDE_DIR "${MEDIAPIPE_DIR}")
        set(MEDIAPIPE_LIB "${MEDIAPIPE_DIR}/bazel-bin/mediapipe/framework/libmediapipe_framework.so")
        set(MEDIAPIPE_FOUND TRUE)
        add_definitions(-DMEDIAPIPE_DIRECT)
    else()
        message(WARNING "Neither libmediapipe nor direct MediaPipe installation found.")
        message(STATUS "Building without MediaPipe support...")
        set(MEDIAPIPE_FOUND FALSE)
        add_definitions(-DMEDIAPIPE_DISABLED)
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
if(MEDIAPIPE_FOUND)
    include_directories(${MEDIAPIPE_INCLUDE_DIR})
    message(STATUS "Found MediaPipe include: ${MEDIAPIPE_INCLUDE_DIR}")
    add_definitions(-DMEDIAPIPE_AVAILABLE)
else()
    message(STATUS "MediaPipe not available, building without MediaPipe support")
    add_definitions(-DMEDIAPIPE_DISABLED)
endif()

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
if(MEDIAPIPE_FOUND)
    target_link_libraries(BloodPressureDLL ${MEDIAPIPE_LIB})
    message(STATUS "Linked MediaPipe: ${MEDIAPIPE_LIB}")
else()
    message(STATUS "MediaPipe not linked (not available)")
endif()

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
