# 血圧推定DLL

30秒WebM動画からrPPGアルゴリズムを使用してRRIを取得し、機械学習モデルで血圧を推定するDLL

## 🎯 概要

- **入力**: WebM動画（30秒、1280x720、30fps）+ 生体パラメータ（身長、体重、性別）
- **処理**: MediaPipe顔検出 → rPPG信号抽出 → RRI計算 → 機械学習による血圧推定
- **出力**: 収縮期/拡張期血圧、PPGローデータ（CSV）、エラー情報

## 📋 要件

- **Python**: 3.12（MediaPipe対応）
- **アーキテクチャ**: 32-bit（要件仕様）
- **OS**: Windows（DLL形式）
- **依存ライブラリ**: OpenCV、MediaPipe、scikit-learn、NumPy、SciPy

## 🚀 セットアップ

### 1. 環境構築

```bash
# Python 3.12の仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 依存パッケージインストール
pip install -r requirements.txt
```

### 2. モデルファイル

`models/`ディレクトリに以下のファイルが必要：
- `model_sbp.pkl` - 収縮期血圧予測モデル
- `model_dbp.pkl` - 拡張期血圧予測モデル

### 3. テスト実行

```bash
# 基本動作テスト
python bp_estimation_dll.py

# DLLインターフェーステスト
python dll_interface.py
```

## 🔧 DLLビルド（Windows）

### Windows 32-bit環境での手順

1. **Python 3.12 (32-bit)をインストール**
2. **依存パッケージインストール**
   ```cmd
   pip install -r requirements.txt
   pip install cx_Freeze
   ```

3. **DLLビルド実行**
   ```cmd
   python build_windows_dll.py build
   ```

4. **生成物確認**
   ```
   build/exe.win32-3.12/
   ├── BloodPressureEstimation.dll  # メインDLL
   ├── BloodPressureEstimation.exe  # スタンドアロン実行ファイル
   ├── models/                      # 学習済みモデル
   ├── lib/                         # 依存ライブラリ
   └── ...
   ```

## 📚 DLL使用方法

### C/C++からの呼び出し

```c
#include <windows.h>

// DLL関数の型定義
typedef BOOL (*InitializeDLLFunc)(const char* model_dir);
typedef int (*StartBPAnalysisFunc)(const char* request_id, int height, int weight, 
                                   int sex, const char* movie_path, void* callback);

// コールバック関数
void OnBPResult(const char* request_id, int sbp, int dbp, 
                const char* csv_data, void* errors) {
    printf("血圧結果: %s - SBP:%d, DBP:%d\\n", request_id, sbp, dbp);
}

int main() {
    // DLLロード
    HINSTANCE hDLL = LoadLibrary(L"BloodPressureEstimation.dll");
    
    // 関数取得
    InitializeDLLFunc InitializeDLL = (InitializeDLLFunc)GetProcAddress(hDLL, "InitializeDLL");
    StartBPAnalysisFunc StartBPAnalysis = (StartBPAnalysisFunc)GetProcAddress(hDLL, "StartBloodPressureAnalysis");
    
    // DLL使用
    if (InitializeDLL("models")) {
        StartBPAnalysis("req_001", 170, 70, 1, "video.webm", OnBPResult);
    }
    
    FreeLibrary(hDLL);
    return 0;
}
```

### Python からの呼び出し

```python
import ctypes

# DLLロード
dll = ctypes.CDLL('./BloodPressureEstimation.dll')

# 関数定義
dll.InitializeDLL.argtypes = [ctypes.c_char_p]
dll.InitializeDLL.restype = ctypes.c_bool

# DLL使用
if dll.InitializeDLL(b"models"):
    print("DLL初期化成功")
```

## 🔍 DLL関数仕様

### 1. InitializeDLL
```c
BOOL InitializeDLL(const char* model_dir);
```
- **機能**: DLLの初期化とモデル読み込み
- **引数**: `model_dir` - モデルディレクトリパス
- **戻り値**: 成功=TRUE, 失敗=FALSE

### 2. StartBloodPressureAnalysis
```c
int StartBloodPressureAnalysis(const char* request_id, int height, int weight, 
                               int sex, const char* movie_path, CallbackFunc callback);
```
- **機能**: 血圧解析の非同期開始
- **引数**:
  - `request_id` - リクエストID（形式: `yyyyMMddHHmmssfff_driverCode`）
  - `height` - 身長（cm）
  - `weight` - 体重（kg）
  - `sex` - 性別（1=男性, 2=女性）
  - `movie_path` - WebM動画ファイルパス
  - `callback` - 結果通知用コールバック関数
- **戻り値**: エラー数（0=成功）

### 3. CancelBloodPressureProcessing
```c
BOOL CancelBloodPressureProcessing(const char* request_id);
```
- **機能**: 指定リクエストIDの処理中断
- **戻り値**: 成功=TRUE, 失敗=FALSE

### 4. GetBloodPressureStatus
```c
const char* GetBloodPressureStatus(const char* request_id);
```
- **機能**: 処理状況取得
- **戻り値**: `"none"` | `"processing"`

### 5. GetDLLVersion
```c
const char* GetDLLVersion();
```
- **機能**: DLLバージョン取得
- **戻り値**: バージョン文字列

## 📞 コールバック仕様

```c
typedef void (*CallbackFunc)(const char* request_id, int sbp, int dbp, 
                             const char* csv_data, ErrorInfo* errors);
```

### パラメータ
- `request_id` - リクエストID
- `sbp` - 収縮期血圧（mmHg、整数）
- `dbp` - 拡張期血圧（mmHg、整数）
- `csv_data` - PPGローデータ（CSV形式、約20KB）
- `errors` - エラー情報配列（NULL=エラーなし）

### ErrorInfo構造体
```c
typedef struct {
    const char* code;        // エラーコード
    const char* message;     // エラーメッセージ
    BOOL is_retriable;       // 再試行可能フラグ
} ErrorInfo;
```

## ⚠️ エラーコード

| コード | 内容 | 備考 |
|--------|------|------|
| 1001 | DLL未初期化 | Init未実行 |
| 1002 | デバイス接続失敗 | カメラ・センサ未接続 |
| 1003 | キャリブレーション未完了 | 実施前に測定要求 |
| 1004 | 入力パラメータ不正 | NULLや異常値 |
| 1005 | 測定中リクエスト不可 | 排他制御タイミング |
| 1006 | DLL内部処理エラー | 想定外の例外発生 |

## 🧪 テスト結果

### サンプル動画での検証結果
- **収縮期血圧**: 126 mmHg
- **拡張期血圧**: 76 mmHg  
- **処理時間**: 約20秒
- **CSVサイズ**: 15,810文字
- **ステータス**: 正常完了

### 処理フロー確認
1. ✅ DLL初期化成功
2. ✅ MediaPipe顔検出動作
3. ✅ rPPG信号抽出完了
4. ✅ 機械学習モデル予測成功
5. ✅ コールバック通知正常
6. ✅ CSVデータ出力完了

## 📝 開発ノート

### 技術スタック
- **顔検出**: MediaPipe FaceMesh
- **信号処理**: POS（Plane Orthogonal to Skin）アルゴリズム
- **機械学習**: Random Forest（scikit-learn）
- **特徴量**: RRI統計値（平均、標準偏差、最小、最大）+ BMI + 性別

### パフォーマンス
- **動画形式**: WebM（VP8）、30秒、30fps、1280x720
- **メモリ使用量**: 約200MB（モデル + 動画バッファ）
- **CPU使用率**: 解析中80-90%（シングルスレッド）

### 制約事項
- 32-bit DLL（要件仕様）
- Windows専用（要件仕様）
- 同時処理不可（排他制御）
- MediaPipe警告あり（動作に影響なし）

## 🔄 今後の改善案

1. **並列処理対応**: 複数リクエスト同時処理
2. **GPU加速**: CUDA対応でパフォーマンス向上
3. **モデル最適化**: ONNX変換でサイズ削減
4. **エラーハンドリング強化**: より詳細なエラー情報
5. **ログ機能**: デバッグ用ログ出力