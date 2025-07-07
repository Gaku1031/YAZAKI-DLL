# 🎯 Windows 10 64bit 血圧推定 DLL 作成ガイド

## 📋 前提条件

### 必要なソフトウェア

1. **Windows 10 64bit**
2. **Python 3.12 (64bit 版)**
3. **Visual Studio 2019/2022 Build Tools**
4. **Git** (オプション)

### システム要件

- **CPU**: Intel/AMD 64bit プロセッサ
- **メモリ**: 8GB 以上推奨
- **ディスク**: 10GB 以上の空き容量
- **OS**: Windows 10 64bit

## 🔧 セットアップ手順

### 1. Python 環境の準備

```bash
# Python 3.12のインストール確認
python --version
# Python 3.12.x が表示されることを確認

# 仮想環境の作成（推奨）
python -m venv venv
venv\Scripts\activate
```

### 2. 必要なパッケージのインストール

```bash
# 基本パッケージのインストール
pip install -r requirements.txt

# DLLビルド用パッケージのインストール
pip install cx_Freeze
pip install pyinstaller  # 代替手段として
```

### 3. Visual Studio Build Tools の確認

```bash
# Visual Studio Build Toolsがインストールされているか確認
where cl.exe
# パスが表示されればOK
```

## 🏗️ DLL 作成手順

### 方法 1: cx_Freeze を使用（推奨）

```bash
# 1. ビルドスクリプトの実行
python build_windows_dll_64bit.py build

# 2. 生成物の確認
dir build\exe.win-amd64-3.12\
```

**生成物:**

- `BloodPressureEstimation.dll` - メイン DLL ファイル
- `BloodPressureEstimation.exe` - スタンドアロン実行ファイル
- 依存ライブラリ（自動的にバンドル）

### 方法 2: PyInstaller を使用（代替）

```bash
# 1. PyInstallerでのビルド
python build_windows_dll_pyinstaller.py

# 2. 生成物の確認
dir dist\
```

**生成物:**

- `BloodPressureEstimation.exe` - DLL として使用可能な実行ファイル

## 📦 配布パッケージの作成

### 最小構成での配布

```bash
# 必要なファイルのみをコピー
mkdir BloodPressureEstimation_DLL
copy build\exe.win-amd64-3.12\BloodPressureEstimation.dll BloodPressureEstimation_DLL\
copy build\exe.win-amd64-3.12\*.dll BloodPressureEstimation_DLL\
copy build\exe.win-amd64-3.12\models\* BloodPressureEstimation_DLL\models\
copy BloodPressureEstimation.h BloodPressureEstimation_DLL\
copy cpp_example.cpp BloodPressureEstimation_DLL\
```

### 配布ファイル構成

```
BloodPressureEstimation_DLL/
├── BloodPressureEstimation.dll     # メインDLL
├── BloodPressureEstimation.h       # C/C++ヘッダーファイル
├── cpp_example.cpp                 # 使用例
├── models/                         # 機械学習モデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
└── README_DLL.md                  # 使用方法
```

## 🔍 DLL の動作確認

### 1. 基本的な動作確認

```bash
# DLLファイルの存在確認
dir BloodPressureEstimation.dll

# 依存ライブラリの確認
dumpbin /dependents BloodPressureEstimation.dll
```

### 2. C++での動作確認

```cpp
#include <windows.h>
#include <iostream>

int main() {
    // DLLロード
    HMODULE hDLL = LoadLibrary(L"BloodPressureEstimation.dll");
    if (!hDLL) {
        std::cout << "DLLロード失敗" << std::endl;
        return -1;
    }

    // 関数取得
    typedef BOOL (*InitFunc)(const char*);
    InitFunc InitializeDLL = (InitFunc)GetProcAddress(hDLL, "InitializeDLL");

    if (InitializeDLL) {
        if (InitializeDLL("models")) {
            std::cout << "DLL初期化成功" << std::endl;
        } else {
            std::cout << "DLL初期化失敗" << std::endl;
        }
    }

    FreeLibrary(hDLL);
    return 0;
}
```

## 🚀 既存システムへの組み込み

### 1. DLL ファイルの配置

```bash
# 既存システムの適切なディレクトリに配置
copy BloodPressureEstimation.dll C:\YourSystem\bin\
copy models\* C:\YourSystem\models\
```

### 2. C/C++プロジェクトでの使用

```cpp
// プロジェクト設定
// 1. インクルードパスに追加
// 2. ライブラリパスに追加
// 3. リンクライブラリに追加

#include "BloodPressureEstimation.h"

// 使用例
void OnBPResult(const char* request_id, int sbp, int dbp,
                const char* csv_data, const BPErrorInfo* errors) {
    printf("血圧結果: %s - SBP:%d, DBP:%d\n", request_id, sbp, dbp);
}

int main() {
    if (InitializeDLL("models")) {
        int error_count = StartBloodPressureAnalysis(
            "test_001", 170, 70, BP_SEX_MALE,
            "video.webm", OnBPResult
        );
    }
    return 0;
}
```

## ⚠️ 注意事項

### 1. アーキテクチャの一致

- **64bit システム**では**64bit DLL**を使用
- **32bit システム**では**32bit DLL**を使用
- アーキテクチャが一致しないとロードエラー

### 2. 依存ライブラリ

- Visual C++ 再頒布可能パッケージが必要
- 必要に応じて Microsoft Visual C++ Redistributable をインストール

### 3. パス設定

- DLL ファイルのパスが正しく設定されていることを確認
- モデルファイルのパスも確認

### 4. 権限設定

- 管理者権限が必要な場合がある
- ファイル書き込み権限の確認

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. DLL ロードエラー

```cpp
// エラー: DLLが見つからない
// 解決: パスを確認、依存ライブラリをインストール
```

#### 2. 関数が見つからない

```cpp
// エラー: GetProcAddressで関数が見つからない
// 解決: 関数名の確認、DLLの再ビルド
```

#### 3. 初期化エラー

```cpp
// エラー: InitializeDLLが失敗
// 解決: モデルファイルのパス確認
```

## 📞 サポート

問題が発生した場合は以下を確認してください：

1. **ログファイル**の確認
2. **依存ライブラリ**の確認
3. **パス設定**の確認
4. **権限設定**の確認

---

**作成者**: IKI Japan/Yazaki  
**バージョン**: 1.0.0  
**最終更新**: 2025-01-06
