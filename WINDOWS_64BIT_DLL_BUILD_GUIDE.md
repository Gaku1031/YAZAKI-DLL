# 🎯 Windows 64bit 血圧推定 DLL 作成ガイド（MediaPipe版）

## 📋 前提条件

### 必要なソフトウェア

1. **Windows 10/11 64bit**
2. **Python 3.12 (64bit 版)**
3. **Visual Studio 2019/2022 Community/Professional**
4. **Git** (オプション)

### システム要件

- **CPU**: Intel/AMD 64bit プロセッサ
- **メモリ**: 16GB 以上推奨
- **ディスク**: 15GB 以上の空き容量
- **OS**: Windows 10/11 64bit
- **GPU**: CUDA対応GPU（オプション、高速化用）

## 🔧 セットアップ手順

### 1. Python 環境の準備

```bash
# Python 3.12 64bit版のインストール確認
python --version
# Python 3.12.x が表示されることを確認

# アーキテクチャの確認
python -c "import platform; print(platform.architecture())"
# ('64bit', 'WindowsPE') が表示されることを確認

# 仮想環境の作成（推奨）
python -m venv venv
venv\Scripts\activate
```

### 2. 必要なパッケージのインストール

```bash
# 基本パッケージのインストール
pip install -r requirements.txt

# DLLビルド用パッケージのインストール
pip install cx_Freeze>=6.15.0

# MediaPipeの動作確認
python -c "import mediapipe as mp; print('MediaPipe OK:', mp.__version__)"
```

### 3. Visual Studio Build Tools の確認

```bash
# Visual Studio Build Toolsがインストールされているか確認
where cl.exe
# パスが表示されればOK

# MSBuildの確認
where msbuild.exe
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

- `BloodPressureEstimation.dll` - メイン 64bit DLL ファイル
- `BloodPressureEstimation.exe` - スタンドアロン実行ファイル
- `python312.dll` - Python ランタイム（64bit）
- `mediapipe/` - MediaPipe関連ファイル
- `models/` - 機械学習モデル
- その他依存ライブラリ（自動的にバンドル）

### 方法 2: 手動ビルド

```bash
# 1. cx_Freezeのセットアップファイルを直接実行
python setup.py build

# 2. カスタムビルドオプション
python -c "
from cx_Freeze import setup, Executable
# カスタムビルド設定
"
```

## 📦 配布パッケージの作成

### 完全版配布パッケージ

```bash
# 配布用ディレクトリの作成
mkdir BloodPressureEstimation_DLL_64bit
cd BloodPressureEstimation_DLL_64bit

# 必要なファイルをコピー
copy ..\build\exe.win-amd64-3.12\BloodPressureEstimation.dll .\
copy ..\build\exe.win-amd64-3.12\*.dll .\
xcopy ..\build\exe.win-amd64-3.12\models .\models\ /E /I
xcopy ..\build\exe.win-amd64-3.12\mediapipe .\mediapipe\ /E /I
copy ..\BloodPressureEstimation.h .\
copy ..\cpp_example.cpp .\
copy ..\README_DLL.md .\
```

### 最小構成での配布

```bash
# 最小限のファイルのみ
mkdir BloodPressureEstimation_DLL_64bit_Minimal
copy BloodPressureEstimation.dll BloodPressureEstimation_DLL_64bit_Minimal\
copy python312.dll BloodPressureEstimation_DLL_64bit_Minimal\
copy mediapipe*.dll BloodPressureEstimation_DLL_64bit_Minimal\
xcopy models\ BloodPressureEstimation_DLL_64bit_Minimal\models\ /E /I
```

### 配布ファイル構成

```
BloodPressureEstimation_DLL_64bit/
├── BloodPressureEstimation.dll     # メイン64bit DLL
├── BloodPressureEstimation.h       # C/C++ヘッダーファイル
├── python312.dll                  # Python ランタイム (64bit)
├── mediapipe/                      # MediaPipe関連ファイル
│   ├── modules/
│   ├── framework/
│   └── python/
├── models/                         # 機械学習モデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
├── lib/                           # 依存ライブラリ
│   ├── numpy.libs/
│   ├── scipy.libs/
│   └── sklearn.libs/
├── cpp_example.cpp                # 使用例
└── README_DLL.md                  # 使用方法
```

## 🔍 DLL の動作確認

### 1. 基本的な動作確認

```bash
# DLLファイルの存在確認
dir BloodPressureEstimation.dll

# DLLの詳細情報
dumpbin /headers BloodPressureEstimation.dll

# 依存ライブラリの確認
dumpbin /dependents BloodPressureEstimation.dll
```

### 2. Python直接テスト

```python
# dll_interface.pyの直接テスト
python dll_interface.py
```

### 3. C++での動作確認

```cpp
#include <windows.h>
#include <iostream>
#include "BloodPressureEstimation.h"

// コールバック関数
void OnBPResult(const char* request_id, int sbp, int dbp, 
                const char* csv_data, const BPErrorInfo* errors) {
    if (errors == nullptr) {
        std::cout << "血圧結果: " << request_id 
                  << " - SBP:" << sbp << ", DBP:" << dbp << std::endl;
        std::cout << "CSVデータサイズ: " << strlen(csv_data) << " 文字" << std::endl;
    } else {
        std::cout << "エラー: " << errors->message << std::endl;
    }
}

int main() {
    // DLLロード
    HMODULE hDLL = LoadLibrary(L"BloodPressureEstimation.dll");
    if (!hDLL) {
        std::cout << "DLLロード失敗: " << GetLastError() << std::endl;
        return -1;
    }

    // 関数取得
    typedef BOOL (*InitFunc)(const char*);
    typedef const char* (*StartFunc)(const char*, int, int, int, const char*, BPAnalysisCallback);
    typedef const char* (*StatusFunc)(const char*);
    typedef const char* (*VersionFunc)(void);

    InitFunc InitializeDLL = (InitFunc)GetProcAddress(hDLL, "InitializeDLL");
    StartFunc StartAnalysis = (StartFunc)GetProcAddress(hDLL, "StartBloodPressureAnalysis");
    StatusFunc GetStatus = (StatusFunc)GetProcAddress(hDLL, "GetBloodPressureStatus");
    VersionFunc GetVersion = (VersionFunc)GetProcAddress(hDLL, "GetDLLVersion");

    if (InitializeDLL && StartAnalysis && GetStatus && GetVersion) {
        // DLL初期化
        if (InitializeDLL("models")) {
            std::cout << "DLL初期化成功" << std::endl;
            std::cout << "DLLバージョン: " << GetVersion() << std::endl;
            
            // 血圧解析実行
            const char* error_code = StartAnalysis(
                "20250709120000123_9000000001_0000012345",
                170, 70, 1,  // 身長170cm, 体重70kg, 男性
                "sample-data\\100万画素.webm",
                OnBPResult
            );
            
            if (error_code == nullptr) {
                std::cout << "血圧解析開始成功" << std::endl;
                
                // 処理完了まで待機
                while (true) {
                    const char* status = GetStatus("20250709120000123_9000000001_0000012345");
                    std::cout << "処理状況: " << status << std::endl;
                    if (strcmp(status, "none") == 0) {
                        break;
                    }
                    Sleep(2000);
                }
            } else {
                std::cout << "血圧解析開始失敗: " << error_code << std::endl;
            }
        } else {
            std::cout << "DLL初期化失敗" << std::endl;
        }
    } else {
        std::cout << "DLL関数取得失敗" << std::endl;
    }

    FreeLibrary(hDLL);
    return 0;
}
```

## 🚀 Visual Studio プロジェクトでの統合

### 1. Visual Studio プロジェクト設定

```cpp
// プロジェクト設定
// プラットフォーム: x64
// 文字セット: Unicode
// ランタイムライブラリ: /MD (DLL版)
```

### 2. インクルードパスとライブラリパスの設定

```
プロジェクト → プロパティ → 構成プロパティ
├── インクルードディレクトリ
│   └── BloodPressureEstimation.h のパス
├── ライブラリディレクトリ  
│   └── BloodPressureEstimation.dll のパス
└── リンカー → 入力
    └── 追加の依存ファイル: （不要、動的ロードのため）
```

### 3. DLLファイルの配置

```
YourProject/
├── x64/
│   └── Debug/ (または Release/)
│       ├── YourApp.exe
│       ├── BloodPressureEstimation.dll
│       ├── python312.dll
│       ├── mediapipe関連DLL
│       └── models/
│           ├── model_sbp.pkl
│           └── model_dbp.pkl
```

### 4. 実用的なC++実装例

```cpp
#pragma once
#include <windows.h>
#include <string>
#include <functional>
#include <mutex>
#include "BloodPressureEstimation.h"

class BloodPressureAnalyzer {
private:
    HMODULE hDLL;
    bool initialized;
    std::mutex mtx;

    // DLL関数ポインタ
    typedef BOOL (*InitFunc)(const char*);
    typedef const char* (*StartFunc)(const char*, int, int, int, const char*, BPAnalysisCallback);
    typedef BOOL (*CancelFunc)(const char*);
    typedef const char* (*StatusFunc)(const char*);
    typedef const char* (*VersionFunc)(void);

    InitFunc InitializeDLL;
    StartFunc StartAnalysis;
    CancelFunc CancelAnalysis;
    StatusFunc GetStatus;
    VersionFunc GetVersion;

public:
    BloodPressureAnalyzer() : hDLL(nullptr), initialized(false) {}
    
    ~BloodPressureAnalyzer() {
        if (hDLL) {
            FreeLibrary(hDLL);
        }
    }

    bool Initialize(const std::string& dllPath = "BloodPressureEstimation.dll",
                   const std::string& modelPath = "models") {
        std::lock_guard<std::mutex> lock(mtx);
        
        // DLLロード
        hDLL = LoadLibraryA(dllPath.c_str());
        if (!hDLL) {
            return false;
        }

        // 関数ポインタ取得
        InitializeDLL = (InitFunc)GetProcAddress(hDLL, "InitializeDLL");
        StartAnalysis = (StartFunc)GetProcAddress(hDLL, "StartBloodPressureAnalysis");
        CancelAnalysis = (CancelFunc)GetProcAddress(hDLL, "CancelBloodPressureProcessing");
        GetStatus = (StatusFunc)GetProcAddress(hDLL, "GetBloodPressureStatus");
        GetVersion = (VersionFunc)GetProcAddress(hDLL, "GetDLLVersion");

        if (!InitializeDLL || !StartAnalysis || !CancelAnalysis || !GetStatus || !GetVersion) {
            FreeLibrary(hDLL);
            hDLL = nullptr;
            return false;
        }

        // DLL初期化
        initialized = InitializeDLL(modelPath.c_str());
        return initialized;
    }

    std::string GetVersionInfo() {
        if (!initialized) return "";
        return GetVersion();
    }

    bool StartAnalysis(const std::string& requestId, int height, int weight, int sex,
                      const std::string& videoPath, BPAnalysisCallback callback) {
        if (!initialized) return false;
        
        const char* error = StartAnalysis(requestId.c_str(), height, weight, sex, 
                                        videoPath.c_str(), callback);
        return (error == nullptr);
    }

    std::string GetAnalysisStatus(const std::string& requestId) {
        if (!initialized) return "error";
        return GetStatus(requestId.c_str());
    }

    bool CancelAnalysis(const std::string& requestId) {
        if (!initialized) return false;
        return CancelAnalysis(requestId.c_str());
    }
};

// 使用例
void TestBloodPressureAnalyzer() {
    BloodPressureAnalyzer analyzer;
    
    if (analyzer.Initialize()) {
        std::cout << "バージョン: " << analyzer.GetVersionInfo() << std::endl;
        
        // コールバック設定
        auto callback = [](const char* id, int sbp, int dbp, const char* csv, const BPErrorInfo* err) {
            if (!err) {
                std::cout << "結果: " << id << " SBP=" << sbp << " DBP=" << dbp << std::endl;
            }
        };
        
        // 解析開始
        if (analyzer.StartAnalysis("test_001", 170, 70, 1, "test_video.webm", callback)) {
            std::cout << "解析開始成功" << std::endl;
        }
    }
}
```

## ⚠️ 注意事項

### 1. アーキテクチャの一致

- **64bit システム**では**64bit DLL**を使用
- **32bit システム**では動作しない
- Visual Studio プロジェクトも **x64** プラットフォームで設定

### 2. 依存ライブラリ

- **Visual C++ 2019/2022 Redistributable (x64)** が必要
- **Python 3.12 Runtime (64bit)** が含まれる
- **MediaPipe関連ライブラリ** が含まれる

### 3. パス設定

- DLL ファイルのパスが正しく設定されていることを確認
- モデルファイル (`models/`) のパスも確認
- 相対パスでの配置を推奨

### 4. メモリ使用量

- MediaPipe使用により **約500MB-1GB** のメモリを使用
- 64bit環境での大容量メモリ活用が可能

### 5. セキュリティ設定

- ウイルス対策ソフトで誤検知される場合がある
- デジタル署名の適用を推奨

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. DLL ロードエラー

```
エラー: LoadLibrary failed
解決: 
- 64bit版DLLを使用しているか確認
- 依存ライブラリ（python312.dll等）が同じフォルダにあるか確認
- Visual C++ Redistributable (x64) をインストール
```

#### 2. MediaPipe 初期化エラー

```
エラー: MediaPipe initialization failed
解決:
- GPU ドライバを最新に更新
- CUDA Runtime のインストール（GPU使用時）
- ウイルス対策ソフトの除外設定
```

#### 3. 関数が見つからない

```
エラー: GetProcAddress failed
解決:
- DLLの関数名を確認（dumpbin /exports）
- DLLが正しくビルドされているか確認
```

#### 4. メモリ不足エラー

```
エラー: Out of memory
解決:
- 64bit環境で実行
- 物理メモリを増設（8GB → 16GB以上）
- 仮想メモリの設定を確認
```

#### 5. 動画読み込みエラー

```
エラー: Video file cannot be opened
解決:
- WebM形式での動画エンコード確認
- ファイルパスに日本語が含まれていないか確認
- ファイルアクセス権限の確認
```

## 📞 サポート

問題が発生した場合は以下を確認してください：

1. **システム要件**の確認
2. **ログファイル**の確認
3. **依存ライブラリ**の確認
4. **パス設定**の確認
5. **権限設定**の確認

## 📈 パフォーマンス最適化

### GPU アクセラレーション

```python
# MediaPipe GPU設定
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=2  # 高精度モード
)
```

### メモリ使用量最適化

```cpp
// C++側でのメモリ管理
class BPAnalysisManager {
private:
    static const int MAX_CONCURRENT_REQUESTS = 4;  // 同時処理数制限
    std::queue<std::string> requestQueue;
    
public:
    bool QueueAnalysis(const std::string& requestId, /* 他のパラメータ */) {
        if (requestQueue.size() >= MAX_CONCURRENT_REQUESTS) {
            return false;  // キューが満杯
        }
        // キューに追加
        requestQueue.push(requestId);
        return true;
    }
};
```

---

**作成者**: IKI Japan/Yazaki  
**バージョン**: 2.0.0 (MediaPipe 64bit版)  
**最終更新**: 2025-01-09