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
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Protobuf（OpenCVの依存関係）
find_package(Protobuf REQUIRED)
if(Protobuf_FOUND)
    include_directories(${Protobuf_INCLUDE_DIRS})
    message(STATUS "Protobuf found: ${Protobuf_VERSION}")
else()
    message(WARNING "Protobuf not found - OpenCV may fail to configure")
endif()

# 個別ライブラリ使用オプション（opencv480_world.dllを不要にする）
option(USE_INDIVIDUAL_OPENCV_LIBS "Use individual OpenCV libraries instead of world" OFF)
if(USE_INDIVIDUAL_OPENCV_LIBS)
    # 通常のOpenCVライブラリを使用
    set(OpenCV_LIBS 
        ${OpenCV_DIR}/lib/opencv_core480.lib
        ${OpenCV_DIR}/lib/opencv_imgproc480.lib
        ${OpenCV_DIR}/lib/opencv_imgcodecs480.lib
        ${OpenCV_DIR}/lib/opencv_objdetect480.lib
        ${OpenCV_DIR}/lib/opencv_dnn480.lib
    )
    message(STATUS "Using individual OpenCV libraries - opencv480_world.dll not required")
    message(STATUS "Estimated size: ~33MB (vs 60MB world)")
else()
    message(STATUS "Using OpenCV world library - opencv480_world.dll required")
endif()

# 軽量版OpenCVオプション（opencv480_world.dllを不要にする）
option(USE_LIGHTWEIGHT_OPENCV "Use lightweight OpenCV (core + imgproc only)" OFF)
if(USE_LIGHTWEIGHT_OPENCV)
    # 必要最小限のOpenCVコンポーネントのみ使用
    set(OpenCV_LIBS 
        ${OpenCV_DIR}/lib/opencv_core480.lib
        ${OpenCV_DIR}/lib/opencv_imgproc480.lib
        ${OpenCV_DIR}/lib/opencv_imgcodecs480.lib
    )
    message(STATUS "Using lightweight OpenCV - opencv480_world.dll not required")
    message(STATUS "Estimated size: ~13MB (vs 60MB world)")
else()
    message(STATUS "Using full OpenCV - opencv480_world.dll required")
endif()

# 静的リンクオプション（opencv480_world.dllを不要にする）
option(USE_STATIC_OPENCV "Use static OpenCV linking" OFF)
if(USE_STATIC_OPENCV)
    # 静的リンクライブラリを使用
    set(OpenCV_LIBS ${OpenCV_LIBS})
    # 静的リンクフラグを追加
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT")
    endif()
    message(STATUS "Using static OpenCV linking - opencv480_world.dll not required")
else()
    message(STATUS "Using dynamic OpenCV linking - opencv480_world.dll required")
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# dlib
find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

# ONNX Runtime
if(DEFINED ONNXRuntime_INCLUDE_DIRS AND DEFINED ONNXRuntime_LIBRARIES)
    # 環境変数から直接設定
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIRS})
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARIES})
    message(STATUS "Using ONNX Runtime from environment variables")
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
    ${dlib_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
)

# ライブラリリンク
target_link_libraries(BloodPressureDLL
    ${OpenCV_LIBS}
    ${dlib_LIBRARIES}
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
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# テストアプリケーション
add_executable(test_app test/test_app.cpp)
target_link_libraries(test_app BloodPressureDLL)

# インストール設定
install(TARGETS BloodPressureDLL
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
    DESTINATION include
)

# dlibモデルファイルのコピー
file(COPY ${CMAKE_SOURCE_DIR}/models/shape_predictor_68_face_landmarks.dat
     DESTINATION ${CMAKE_BINARY_DIR}/bin) 
