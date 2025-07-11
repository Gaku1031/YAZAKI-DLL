# C#から呼び出し可能な64bit DLLビルド手順

## 概要
Visual Studio 2022とPyInstallerを使用して、C#から呼び出し可能な64bit血圧推定DLLを作成します。

## 前提条件
- Windows 10/11 64bit
- Visual Studio 2022 (Community版以上)
- Python 3.8以上 (64bit版)
- Git

## 手順

### 1. 環境準備

```bash
# リポジトリクローン
git clone <repository-url>
cd YAZAKI-DLL

# Python仮想環境作成（64bit確認）
python -c "import platform; print(f'Architecture: {platform.architecture()}')"
python -m venv venv64
venv64\Scripts\activate

# 依存関係インストール
pip install -r requirements_balanced_20mb.txt
```

### 2. DLL作成

```bash
# 修正済みスクリプト実行
python build_balanced_20mb_dll.py
```

このスクリプトは以下を実行します：
- C#エクスポート対応のPythonコード生成
- DLL形式のPyInstaller spec作成
- 64bitビルド実行
- `dist/BloodPressureEstimation.dll` 生成

### 3. エクスポート関数確認

```bash
# Visual Studio Developer Command Promptで実行
dumpbin /exports dist\BloodPressureEstimation.dll
```

期待される出力：
```
InitializeDLL
StartBloodPressureAnalysisRequest
GetProcessingStatus
CancelBloodPressureAnalysis
GetVersionInfo
```

### 4. C#テスト実行

```bash
# C#テストコードコンパイル
csc CSharpDllTest.cs

# テスト実行
CSharpDllTest.exe
```

## 変更点

### Python側の修正

1. **エクスポート関数の定義方法変更**
   - `@ctypes.WINFUNCTYPE` デコレータ削除
   - 単純な関数定義に変更
   - `argtypes`/`restype`属性による型定義追加

2. **PyInstaller設定変更**
   - `EXE` から `SHARED` ビルドに変更
   - DLL形式での出力設定

3. **エラーハンドリング強化**
   - C#からの呼び出し時のエラー情報改善
   - デバッグ情報追加

### C#側の対応

1. **DLLImport設定**
   - `CallingConvention.Cdecl` 使用
   - `CharSet.Ansi` 指定
   - 適切な`MarshalAs`属性設定

2. **エラーハンドリング**
   - `DllNotFoundException`
   - `EntryPointNotFoundException`
   - 詳細なエラーメッセージ表示

## トラブルシューティング

### エントリポイントが見つからない場合

1. DLLのエクスポート確認：
   ```bash
   dumpbin /exports BloodPressureEstimation.dll
   ```

2. 64bit/32bit一致確認：
   ```csharp
   Console.WriteLine($"Process: {Environment.Is64BitProcess}");
   Console.WriteLine($"OS: {Environment.Is64BitOperatingSystem}");
   ```

3. DEF ファイル使用（必要に応じて）：
   ```
   EXPORTS
   InitializeDLL
   StartBloodPressureAnalysisRequest
   GetProcessingStatus
   CancelBloodPressureAnalysis
   GetVersionInfo
   ```

### サイズ最適化

- 目標: 20MB以下
- 現在の除外設定で約15-25MB
- 必要に応じて`EXCLUDED_MODULES`を追加

### 依存関係問題

```bash
# 必要最小限のパッケージ確認
pip freeze > current_requirements.txt
```

## ファイル構成

```
YAZAKI-DLL/
├── build_balanced_20mb_dll.py          # メインビルドスクリプト
├── bp_estimation_balanced_20mb.py      # 生成されるPythonDLL
├── BloodPressureEstimation.def         # DLLエクスポート定義
├── BloodPressureEstimation_Balanced20MB.spec  # PyInstaller設定
├── CSharpDllTest.cs                    # C#テストコード
├── requirements_balanced_20mb.txt      # Python依存関係
├── dist/
│   └── BloodPressureEstimation.dll     # 生成されるDLL
└── models/                             # 機械学習モデル（オプション）
```

## 注意事項

1. **64bit環境必須**: Python、Visual Studio、ターゲットアプリすべて64bit
2. **依存DLL**: 必要に応じて Visual C++ Redistributable
3. **パス設定**: DLLは実行ファイルと同じディレクトリか、PATHに配置
4. **モデルディレクトリ**: `models/`フォルダも配布に含める

## 成功の確認

C#テストで以下が表示されれば成功：
```
=== 血圧推定DLL C#テスト開始 ===
1. DLL初期化
   初期化結果: True
2. バージョン情報取得
   バージョン: v1.0.0-balanced-20mb
...
=== DLLテスト完了 ===
```