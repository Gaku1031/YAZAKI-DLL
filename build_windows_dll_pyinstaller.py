"""
Windows 64-bit DLLビルドスクリプト（PyInstaller版）
PyInstallerを使用してPythonコードをWindows 64bit DLLとしてコンパイル
"""

import PyInstaller.__main__
import os
import sys

def build_dll_with_pyinstaller():
    """PyInstallerを使用してDLLをビルド"""
    
    # PyInstallerのオプション
    options = [
        'dll_interface.py',  # メインファイル
        '--onefile',  # 単一ファイルにまとめる
        '--windowed',  # コンソールウィンドウを表示しない
        '--name=BloodPressureEstimation',  # 出力ファイル名
        '--distpath=dist',  # 出力ディレクトリ
        '--workpath=build',  # 作業ディレクトリ
        '--specpath=build',  # specファイルの場所
        '--clean',  # ビルド前にクリーンアップ
        '--noconfirm',  # 確認なしで上書き
        '--hidden-import=cv2',
        '--hidden-import=mediapipe',
        '--hidden-import=numpy',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=pandas',
        '--hidden-import=joblib',
        '--hidden-import=pywt',
        '--hidden-import=threading',
        '--hidden-import=ctypes',
        '--hidden-import=logging',
        '--hidden-import=collections',
        '--hidden-import=datetime',
        '--hidden-import=time',
        '--add-data=models;models',  # モデルファイルを含める
        '--add-data=sample-data;sample-data',  # サンプルデータを含める
        '--add-data=bp_estimation_dll.py;.',  # メインモジュールを含める
    ]
    
    # PyInstallerを実行
    PyInstaller.__main__.run(options)
    
    print("""
=== PyInstaller DLL ビルド完了 ===

生成物:
- dist/BloodPressureEstimation.exe (DLLとして使用可能)

使用方法:
1. dist/BloodPressureEstimation.exe を DLL として使用
2. LoadLibrary() でロード
3. GetProcAddress() で関数を取得

注意:
- .exeファイルですが、DLLとして機能します
- 依存ライブラリは自動的にバンドルされます
- モデルファイルも含まれています
""")

if __name__ == "__main__":
    build_dll_with_pyinstaller() 
