"""
テスト環境セットアップスクリプト
DLLテストに必要な環境を準備
"""

import os
import sys
from pathlib import Path

def create_models_directory():
    """modelsディレクトリとダミーモデルファイル作成"""
    print("=== modelsディレクトリセットアップ ===")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # ダミーSBPモデル作成
    sbp_model_code = '''import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# ダミーSBPモデル作成
model = RandomForestRegressor(n_estimators=10, random_state=42)
X_dummy = np.random.rand(100, 6)  # 特徴量6個
y_dummy = np.random.randint(90, 180, 100)  # SBP範囲
model.fit(X_dummy, y_dummy)

# モデル保存
joblib.dump(model, "models/model_sbp.pkl")
print("✓ SBPモデル作成完了")
'''
    
    # ダミーDBPモデル作成
    dbp_model_code = '''import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# ダミーDBPモデル作成
model = RandomForestRegressor(n_estimators=10, random_state=42)
X_dummy = np.random.rand(100, 6)  # 特徴量6個
y_dummy = np.random.randint(60, 110, 100)  # DBP範囲
model.fit(X_dummy, y_dummy)

# モデル保存
joblib.dump(model, "models/model_dbp.pkl")
print("✓ DBPモデル作成完了")
'''
    
    try:
        # SBPモデル作成
        exec(sbp_model_code)
        
        # DBPモデル作成
        exec(dbp_model_code)
        
        print("✓ ダミーモデルファイル作成完了")
        return True
        
    except Exception as e:
        print(f"✗ モデル作成エラー: {e}")
        print("scikit-learnとjoblibが必要です")
        return False

def check_dependencies():
    """依存関係チェック"""
    print("\n=== 依存関係チェック ===")
    
    required_packages = [
        'opencv-python',
        'mediapipe', 
        'numpy',
        'scikit-learn',
        'joblib',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✓ {package}: {cv2.__version__}")
            elif package == 'mediapipe':
                import mediapipe as mp
                print(f"✓ {package}: {mp.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"✓ {package}: {np.__version__}")
            elif package == 'scikit-learn':
                import sklearn
                print(f"✓ {package}: {sklearn.__version__}")
            elif package == 'joblib':
                import joblib
                print(f"✓ {package}: {joblib.__version__}")
            elif package == 'scipy':
                import scipy
                print(f"✓ {package}: {scipy.__version__}")
        except ImportError:
            print(f"✗ {package}: 未インストール")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n必要パッケージをインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_sample_video():
    """サンプル動画確認"""
    print("\n=== サンプル動画確認 ===")
    
    sample_video = Path("sample-data") / "100万画素.webm"
    
    if sample_video.exists():
        size_mb = sample_video.stat().st_size / (1024 * 1024)
        print(f"✓ サンプル動画確認: {sample_video}")
        print(f"  サイズ: {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ サンプル動画が見つかりません: {sample_video}")
        print("sample-data/ディレクトリに100万画素.webmファイルが必要です")
        return False

def check_dll_files():
    """DLL関連ファイル確認"""
    print("\n=== DLL関連ファイル確認 ===")
    
    required_files = [
        "bp_estimation_facemesh_spec_compliant.py",
        "build_spec_compliant_dll.py",
        "test_dll_with_sample_video.py"
    ]
    
    all_present = True
    
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} が見つかりません")
            all_present = False
    
    return all_present

def main():
    """メインセットアップ処理"""
    print("血圧推定DLL テスト環境セットアップ")
    print("=" * 50)
    
    setup_results = {}
    
    # 1. 依存関係チェック
    setup_results['dependencies'] = check_dependencies()
    
    # 2. modelsディレクトリとモデルファイル作成
    if setup_results['dependencies']:
        setup_results['models'] = create_models_directory()
    else:
        setup_results['models'] = False
        print("依存関係が不足しているため、モデル作成をスキップします")
    
    # 3. サンプル動画確認
    setup_results['sample_video'] = check_sample_video()
    
    # 4. DLL関連ファイル確認
    setup_results['dll_files'] = check_dll_files()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("セットアップ結果")
    print("=" * 50)
    
    all_ready = True
    for check_name, result in setup_results.items():
        status = "✓ OK" if result else "✗ NG"
        print(f"{check_name}: {status}")
        if not result:
            all_ready = False
    
    if all_ready:
        print("\n🎉 テスト環境セットアップ完了！")
        print("\n次の手順:")
        print("1. python build_spec_compliant_dll.py でDLLビルド")
        print("2. python test_dll_with_sample_video.py でテスト実行")
    else:
        print("\n❌ セットアップが完了していません")
        print("上記のエラーを修正してから再実行してください")
    
    return all_ready

if __name__ == "__main__":
    main()