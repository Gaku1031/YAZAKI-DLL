# 32bit DLL作成・デプロイメント完全ガイド

## 🖥️ 必要な環境

### Windows 32bit環境
1. **Windows 10/11 (32bit) または Windows 32bit VM**
2. **Python 3.12.0 (32bit版)**
   - 📥 [python.org](https://www.python.org/downloads/release/python-3120/) から「Windows installer (32-bit)」をダウンロード
   - ⚠️ 64bit版ではなく、必ず32bit版をインストール

### 環境確認コマンド
```cmd
python -c "import platform; print(f'Architecture: {platform.architecture()}')"
```
出力例：`Architecture: ('32bit', 'WindowsPE')`

## 🔧 ステップ1: 開発環境セットアップ

### 1.1 プロジェクトファイルの転送
現在のmacOS環境から以下のファイルをWindows環境に転送：

```
YAZAKI-DLL/
├── bp_estimation_dll.py          # メインDLL実装
├── dll_interface.py              # DLLインターフェース
├── build_windows_dll.py          # ビルドスクリプト
├── requirements.txt              # 依存パッケージ
├── models/                       # 学習済みモデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
└── sample-data/                  # テスト用動画
    └── 100万画素.webm
```

### 1.2 仮想環境作成（Windows）
```cmd
cd YAZAKI-DLL
python -m venv venv_32bit
venv_32bit\Scripts\activate
```

### 1.3 依存パッケージインストール
```cmd
pip install --upgrade pip
pip install -r requirements.txt
pip install cx_Freeze
```

## 🏗️ ステップ2: DLLビルド実行

### 2.1 ビルドコマンド
```cmd
python build_windows_dll.py build
```

### 2.2 生成される構造
```
build/exe.win32-3.12/
├── BloodPressureEstimation.dll   # 🎯 メインDLL
├── BloodPressureEstimation.exe   # テスト用実行ファイル
├── python312.dll               # Python ランタイム
├── _ssl.pyd                     # SSL サポート
├── _socket.pyd                  # ソケット サポート
├── models/                      # 学習済みモデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
├── lib/                         # 依存ライブラリ
│   ├── numpy/
│   ├── opencv/
│   ├── mediapipe/
│   ├── scipy/
│   ├── sklearn/
│   └── ...
└── mediapipe/                   # MediaPipe データ
    └── modules/
```

## 📦 ステップ3: 最終配布パッケージ作成

### 3.1 配布用フォルダ構成
```
BloodPressureEstimationDLL_v1.0/
├── bin/                         # 実行ファイル
│   ├── BloodPressureEstimation.dll
│   ├── python312.dll
│   └── *.pyd files
├── models/                      # 学習済みモデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
├── lib/                         # 依存ライブラリ
├── include/                     # ヘッダーファイル
│   └── BloodPressureEstimation.h
├── examples/                    # サンプルコード
│   ├── cpp_example.cpp
│   ├── c_example.c
│   └── python_example.py
└── docs/                        # ドキュメント
    ├── API_Reference.md
    └── Integration_Guide.md
```

### 3.2 C/C++ヘッダーファイル作成