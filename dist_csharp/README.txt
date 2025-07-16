# BloodPressureEstimation C# Integration Package (C++ラッパーDLL方式)

## パッケージ内容
- CppWrapper.dll : C++ラッパーDLL（C#から呼び出し）
- python311.dll : Pythonランタイム
- BloodPressureEstimation.pyd : Cython DLL
- python_deps/ : NumPy, OpenCV, SciPy, sklearn等の依存
- models/ : 学習済みモデル
- BloodPressureTest.exe : C#テストアプリ

## 使い方
1. C#からDllImportで`CppWrapper.dll`の関数を呼び出す
2. CppWrapper.dllがPythonランタイムを初期化し、Cython pydを呼び出す

### C#からの呼び出し例
```csharp
[DllImport("CppWrapper.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
public static extern int InitializeBP([MarshalAs(UnmanagedType.LPStr)] string modelDir);

int result = InitializeBP("models");
```

## 注意事項
- python311.dll, BloodPressureEstimation.pyd, python_deps/ などが同じディレクトリに必要です。
- Python依存のパスやDLLロード順序に注意してください。
- 必要に応じてPATH, PYTHONPATH, PYTHONHOME等の環境変数を設定してください。 
