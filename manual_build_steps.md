# C++ Wrapper DLL 手動ビルド手順

## 前提条件確認

### 1. Visual Studio 2022の確認
```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

もしパスが異なる場合は以下で確認：
```cmd
dir "C:\Program Files\Microsoft Visual Studio" /s /b | findstr vcvars64.bat
```

### 2. CMakeの確認
```cmd
cmake --version
```

## 手動ビルド手順

### ステップ1: 環境設定
```cmd
# Visual Studio環境設定
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### ステップ2: ビルドディレクトリ作成
```cmd
# 既存のbuildフォルダがあれば削除
if exist build rmdir /s /q build

# 新しいbuildフォルダ作成
mkdir build
cd build
```

### ステップ3: CMake設定
```cmd
# CMakeでVisual Studio 2022プロジェクト生成
cmake .. -G "Visual Studio 17 2022" -A x64
```

### ステップ4: ビルド実行
```cmd
# Release設定でビルド
cmake --build . --config Release
```

### ステップ5: 結果確認
```cmd
# DLLファイル確認
if exist Release\BloodPressureEstimation.dll (
    echo ✓ DLL作成成功
    
    # distフォルダにコピー
    if not exist dist mkdir dist
    copy Release\BloodPressureEstimation.dll dist\
    
    # エクスポート関数確認
    dumpbin /exports dist\BloodPressureEstimation.dll
) else (
    echo ✗ DLL作成失敗
    echo エラーログを確認してください
)
```

## トラブルシューティング

### エラー1: vcvars64.batが見つからない
```cmd
# Visual Studio 2022のインストール確認
dir "C:\Program Files\Microsoft Visual Studio\2022" /b
```

Community版がない場合：
- Professional版: `\Professional\VC\Auxiliary\Build\vcvars64.bat`
- Enterprise版: `\Enterprise\VC\Auxiliary\Build\vcvars64.bat`

### エラー2: Python開発ヘッダーが見つからない
```cmd
# Python開発パッケージ確認
python -c "import sysconfig; print(sysconfig.get_path('include'))"
python -c "import sysconfig; print(sysconfig.get_path('stdlib'))"
```

### エラー3: CMakeでPythonが見つからない
```cmd
# Python実行可能ファイルのパス確認
where python
```

CMakeLists.txtでパス指定：
```cmake
# 明示的なPython指定
set(Python3_EXECUTABLE "C:/Users/mitsu_ubnutu/source/repos/Gaku1031/YAZAKI-DLL/venv/Scripts/python.exe")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
```

## ファイル構成確認

ビルド成功後の構成：
```
YAZAKI-DLL/
├── build/
│   ├── Release/
│   │   └── BloodPressureEstimation.dll
│   └── dist/
│       └── BloodPressureEstimation.dll
├── BloodPressureEstimation.cpp
├── BloodPressureEstimation.h
├── BloodPressureEstimation.def
├── bp_estimation_simple.py
└── CMakeLists.txt
```

## 成功確認

### 1. DLLエクスポート確認
```cmd
dumpbin /exports build\dist\BloodPressureEstimation.dll
```

期待される出力：
```
Exports
   ordinal    name
         1    InitializeDLL
         2    StartBloodPressureAnalysisRequest
         3    GetProcessingStatus
         4    CancelBloodPressureAnalysis
         5    GetVersionInfo
```

### 2. 依存関係確認
```cmd
dumpbin /dependents build\dist\BloodPressureEstimation.dll
```

Python DLLが含まれていることを確認。

## 次のステップ

DLLビルド成功後：

### 1. C#テスト準備
```cmd
# テスト用ディレクトリ作成
mkdir test_csharp
cd test_csharp

# 必要ファイルコピー
copy ..\build\dist\BloodPressureEstimation.dll .
copy ..\bp_estimation_simple.py .
copy ..\CSharpCppWrapperTest.cs .
```

### 2. C#テストコンパイル・実行
```cmd
# C#コンパイル
csc CSharpCppWrapperTest.cs

# テスト実行
CSharpCppWrapperTest.exe
```

期待される出力：
```
=== C++ Wrapper DLL テスト ===
1. DLL初期化
   結果: True
2. バージョン取得
   バージョン: v1.0.0-cpp-wrapper
...
=== テスト完了 ===
```