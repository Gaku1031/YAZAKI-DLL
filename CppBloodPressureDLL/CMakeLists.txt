cmake_minimum_required(VERSION 3.15)
project(BloodPressureDLL)

# C++標準を設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイラ設定
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# GitHub Actions用の最適化設定
option(USE_GITHUB_ACTIONS_OPTIMIZED "Use optimized settings for GitHub Actions" OFF)
if(USE_GITHUB_ACTIONS_OPTIMIZED)
    # GitHub Actions用の最適化設定
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
    message(STATUS "GitHub Actions optimization enabled")
    # 個別ライブラリ使用を有効化
    set(USE_INDIVIDUAL_OPENCV_LIBS ON)
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# vcpkgでインストールされたOpenCVの情報を表示
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")

# vcpkgでインストールされたOpenCVライブラリのパスを確認
if(OpenCV_LIBS)
    message(STATUS "OpenCV libraries found:")
    foreach(lib ${OpenCV_LIBS})
        message(STATUS "  ${lib}")
    endforeach()
else()
    message(WARNING "OpenCV_LIBS is empty")
endif()

# Protobuf（OpenCVの依存関係）
find_package(Protobuf REQUIRED)
if(Protobuf_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    message(STATUS "Protobuf found: ${Protobuf_VERSION}")
else()
    message(WARNING "Protobuf not found - OpenCV may fail to configure")
endif()

# 個別ライブラリ使用オプション
option(USE_INDIVIDUAL_OPENCV_LIBS "Use individual OpenCV libraries instead of world" OFF)

# OpenCV設定の整理
if(USE_INDIVIDUAL_OPENCV_LIBS)
    message(STATUS "Using individual OpenCV libraries from vcpkg")
    message(STATUS "Estimated size: ~33MB (vs 60MB world)")
else()
    message(STATUS "Using OpenCV world library - opencv480_world.dll required")
endif()

# 静的リンクオプション
if(USE_STATIC_OPENCV)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
    endif()
    message(STATUS "Using static OpenCV linking")
else()
    message(STATUS "Using dynamic OpenCV linking")
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# dlib
find_package(dlib REQUIRED)

# ONNX Runtime
if(DEFINED ONNXRUNTIME_INCLUDE_DIRS AND DEFINED ONNXRUNTIME_LIBRARIES)
    # 環境変数から直接設定
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIRS})
    set(ONNXRuntime_LIBRARIES ${ONNXRUNTIME_LIBRARIES})
    message(STATUS "Using ONNX Runtime from environment variables")
    message(STATUS "ONNXRuntime_INCLUDE_DIRS: ${ONNXRuntime_INCLUDE_DIRS}")
    message(STATUS "ONNXRuntime_LIBRARIES: ${ONNXRuntime_LIBRARIES}")
    
    # ヘッダーファイルの存在確認
    if(EXISTS "${ONNXRuntime_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime header file found")
    else()
        message(FATAL_ERROR "ONNX Runtime header file not found at: ${ONNXRuntime_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    endif()
elseif(DEFINED ONNXRUNTIME_ROOT)
    # ONNXRUNTIME_ROOTから手動で設定
    set(ONNXRuntime_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    set(ONNXRuntime_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_ROOT: ${ONNXRUNTIME_ROOT}")
    message(STATUS "ONNXRuntime_INCLUDE_DIRS: ${ONNXRuntime_INCLUDE_DIRS}")
    message(STATUS "ONNXRuntime_LIBRARIES: ${ONNXRuntime_LIBRARIES}")
    
    # ヘッダーファイルの存在確認
    if(EXISTS "${ONNXRuntime_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime header file found")
    else()
        message(FATAL_ERROR "ONNX Runtime header file not found at: ${ONNXRuntime_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    endif()
else()
    # 従来のfind_packageを使用
    find_package(ONNXRuntime REQUIRED)
endif()
include_directories(${ONNXRuntime_INCLUDE_DIRS})

# ソースファイル
set(SOURCES
    src/BloodPressureDLL.cpp
    src/rppg.cpp
    src/bp_estimator.cpp
    src/peak_detect.cpp
)

# ヘッダーファイル
set(HEADERS
    include/BloodPressureDLL.h
    src/rppg.h
    src/bp_estimator.h
    src/peak_detect.h
)

# ライブラリ作成
add_library(BloodPressureDLL SHARED ${SOURCES} ${HEADERS})

# インクルードディレクトリ設定
target_include_directories(BloodPressureDLL PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRuntime_INCLUDE_DIRS}
)

# ライブラリリンク
target_link_libraries(BloodPressureDLL
    ${OpenCV_LIBS}
    dlib::dlib
    ${ONNXRuntime_LIBRARIES}
)

# 静的リンクの場合の追加設定
if(USE_STATIC_OPENCV)
    # 静的リンク時の追加ライブラリ
    if(WIN32)
        target_link_libraries(BloodPressureDLL
            ws2_32
            wininet
            crypt32
            psapi
            ole32
            oleaut32
            uuid
            gdi32
            advapi32
            user32
            kernel32
        )
    endif()
endif()

# コンパイル定義
target_compile_definitions(BloodPressureDLL PRIVATE
    BLOOD_PRESSURE_DLL_EXPORTS
    _CRT_SECURE_NO_WARNINGS
)

# 出力ディレクトリ
set_target_properties(BloodPressureDLL PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# テストアプリケーション
add_executable(test_app test/test_app.cpp)
target_link_libraries(test_app BloodPressureDLL)

# テストアプリケーションのインクルードディレクトリ設定
target_include_directories(test_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# テストアプリケーションの出力ディレクトリ設定
set_target_properties(test_app PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# インストール設定
install(TARGETS BloodPressureDLL
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include
)

# dlibモデルファイルのコピー（存在する場合のみ）
if(EXISTS ${CMAKE_SOURCE_DIR}/models/shape_predictor_68_face_landmarks.dat)
    file(COPY ${CMAKE_SOURCE_DIR}/models/shape_predictor_68_face_landmarks.dat
         DESTINATION ${CMAKE_BINARY_DIR}/bin)
    message(STATUS "dlib face landmarks file copied")
else()
    message(WARNING "dlib face landmarks file not found: ${CMAKE_SOURCE_DIR}/models/shape_predictor_68_face_landmarks.dat")
    message(STATUS "This file is optional for dlib face detection")
endif() 
