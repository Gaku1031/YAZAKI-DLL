# DLL読み込みエラー修正ガイド

## エラー症状
```
DLL初期化エラー: DLL 'BloodPressureEstimation.dll' を読み込めません: 指定されたモジュールが見つかりません。 (HRESULT からの例外:0x8007007E)
```

## 原因
BloodPressureEstimation.dllがPython DLLに依存しているが、Python DLLが見つからない。

## 🚀 クイック修正（推奨）

### 1. test_environmentディレクトリで実行
```cmd
cd test_environment
..\fix_dll_dependencies.bat
```

### 2. 手動修正（上記がうまくいかない場合）

#### Step 1: Python DLLの場所確認
```cmd
python -c "import sys; import os; print(os.path.dirname(sys.executable) + '\\python311.dll')"
```

#### Step 2: Python DLLをコピー
```cmd
# 上記で表示されたパスからコピー
copy "C:\Users\...\python311.dll" .
```

#### Step 3: 追加DLLをコピー
```cmd
# Python実行ディレクトリから
copy "C:\Users\...\vcruntime140.dll" .
copy "C:\Users\...\vcruntime140_1.dll" .
copy "C:\Users\...\msvcp140.dll" .
```

## 🔍 診断ツール

問題の詳細を確認したい場合：
```cmd
cd test_environment
..\diagnose_dll.bat
```

## 💡 具体的な修正手順

### あなたの環境での修正手順

1. **test_environmentディレクトリで以下を実行**:

```cmd
# Python DLLのパス確認
python -c "import sys; import os; print('Python DLL:', os.path.dirname(sys.executable) + '\\python311.dll')"

# 表示されたパスのDLLをコピー（例）
copy "C:\Users\mitsu_ubnutu\source\repos\Gaku1031\YAZAKI-DLL\venv\python311.dll" .

# 追加で必要なDLL（Python実行ディレクトリから）
copy "C:\Users\mitsu_ubnutu\source\repos\Gaku1031\YAZAKI-DLL\venv\vcruntime140.dll" .
copy "C:\Users\mitsu_ubnutu\source\repos\Gaku1031\YAZAKI-DLL\venv\vcruntime140_1.dll" .
```

2. **コピー後、再度テスト実行**:
```cmd
BloodPressureTest.exe
```

## 🎯 期待される結果

修正後は以下のように表示されるはずです：
```
=== 血圧推定DLL実動テスト ===

1. 環境確認
✓ BloodPressureEstimation.dll 確認
✓ bp_estimation_simple.py 確認
✓ サンプル動画確認: sample-data\100万画素.webm

2. DLL初期化
✓ DLL初期化成功
バージョン: v1.0.0-cpp-wrapper
```

## 🔧 代替案

### Visual C++ Redistributableのインストール
問題が解決しない場合：
1. [Microsoft Visual C++ 2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) をダウンロード
2. インストール後、PCを再起動
3. テストを再実行

### Python仮想環境の問題
venv環境を使用している場合：
```cmd
# メインのPython環境を確認
where python

# システムのPythonからDLLをコピー
copy "C:\Program Files\Python311\python311.dll" .
```

## 📋 トラブルシューティング

### Q: fix_dll_dependencies.batが動かない
**A**: 手動でPython DLLをコピーしてください

### Q: どのDLLをコピーすべきかわからない
**A**: diagnose_dll.batを実行して依存関係を確認

### Q: それでも動かない
**A**: SimpleBloodPressureTest.exe（シンプル版）を試してください

## ⚠️ 重要な注意

- 64bit版のPython DLLが必要です
- Python 3.11の場合は `python311.dll`
- venv環境では元のPython環境のDLLが必要な場合があります