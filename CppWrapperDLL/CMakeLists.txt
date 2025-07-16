cmake_minimum_required(VERSION 3.15)
project(CppWrapperDLL LANGUAGES CXX)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_STANDARD 17)

# Pythonインストールパスの自動検出（必要に応じて修正）
find_path(PYTHON_INCLUDE_DIR Python.h PATHS $ENV{PYTHONHOME}/include)
find_library(PYTHON_LIBRARY python311 PATHS $ENV{PYTHONHOME}/libs $ENV{PYTHONHOME})

add_library(CppWrapperDLL SHARED CppWrapperDLL.cpp)

target_include_directories(CppWrapperDLL PRIVATE ${PYTHON_INCLUDE_DIR})
target_link_libraries(CppWrapperDLL PRIVATE ${PYTHON_LIBRARY})

# DLL名を明示
set_target_properties(CppWrapperDLL PROPERTIES OUTPUT_NAME "CppWrapper") 
