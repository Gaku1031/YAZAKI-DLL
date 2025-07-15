# CppWrapperDLL

C#アプリケーションから Cython/Python コードを安全に呼び出すための C++ラッパー DLL です。

## 構成

- CppWrapper.dll（本 DLL）
- python311.dll（Python ランタイム）
- BloodPressureEstimation.pyd（Cython DLL）
- python_deps/（NumPy, OpenCV 等の依存）
- models/（学習済みモデル）

## 使い方

1. C#から DllImport で`CppWrapper.dll`の関数を呼び出す
2. CppWrapper.dll が Python ランタイムを初期化し、Cython pyd を呼び出す

## C#からの呼び出し例

```csharp
[DllImport("CppWrapper.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
public static extern int InitializeBP([MarshalAs(UnmanagedType.LPStr)] string modelDir);

// 使用例
int result = InitializeBP("models");
```

## ビルド方法（Windows, CMake）

```sh
cd CppWrapperDLL
mkdir build
cd build
cmake .. -DPYTHONHOME="C:/path/to/python311"  # 必要に応じてパス指定
cmake --build . --config Release
```

## 注意

- python311.dll, BloodPressureEstimation.pyd, python_deps/ などが同じディレクトリに必要です。
- Python 依存のパスや DLL ロード順序に注意してください。
