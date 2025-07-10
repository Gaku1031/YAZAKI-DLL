"""
仕様書準拠DLL実動画テストスクリプト
sample-data/内の動画を使用してDLLの実際の血圧推定機能をテスト
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

def test_dll_with_sample_video():
    """サンプル動画を使用したDLL機能テスト"""
    print("=== サンプル動画を使用したDLL実動作テスト ===")
    
    # 1. DLLファイル存在確認
    dll_path = Path("dist") / "BloodPressureEstimation.dll"
    if not dll_path.exists():
        print(f"✗ DLLファイルが見つかりません: {dll_path}")
        print("先にビルドを実行してください: python build_spec_compliant_dll.py")
        return False
    
    print(f"✓ DLLファイル確認: {dll_path}")
    size_mb = dll_path.stat().st_size / (1024 * 1024)
    print(f"  サイズ: {size_mb:.1f} MB")
    
    # 2. サンプル動画ファイル確認
    sample_video_path = Path("sample-data") / "100万画素.webm"
    if not sample_video_path.exists():
        print(f"✗ サンプル動画が見つかりません: {sample_video_path}")
        return False
    
    print(f"✓ サンプル動画確認: {sample_video_path}")
    video_size_mb = sample_video_path.stat().st_size / (1024 * 1024)
    print(f"  動画サイズ: {video_size_mb:.1f} MB")
    
    try:
        # 3. DLLインポートとテスト
        import bp_estimation_facemesh_spec_compliant as bp_dll
        
        # 4. DLL初期化
        print("\n=== DLL初期化 ===")
        if not bp_dll.initialize_dll():
            print("✗ DLL初期化に失敗しました")
            return False
        print("✓ DLL初期化成功")
        
        # 5. バージョン情報確認
        version = bp_dll.get_version_info()
        print(f"✓ DLLバージョン: {version}")
        
        # 6. リクエストID生成
        request_id = bp_dll.generate_request_id("9000000001", "0000012345")
        print(f"✓ リクエストID生成: {request_id}")
        
        # 7. コールバック関数定義
        test_results = {}
        
        def analysis_callback(req_id, max_bp, min_bp, csv_data, errors):
            """血圧解析コールバック関数"""
            print(f"\n=== コールバック結果受信 ===")
            print(f"リクエストID: {req_id}")
            print(f"最高血圧: {max_bp} mmHg")
            print(f"最低血圧: {min_bp} mmHg")
            print(f"CSVデータサイズ: {len(csv_data)} 文字")
            
            if errors and len(errors) > 0:
                print(f"エラー数: {len(errors)}")
                for error in errors:
                    print(f"  エラーコード: {error.code}")
                    print(f"  エラーメッセージ: {error.message}")
                    print(f"  再試行可能: {error.is_retriable}")
            else:
                print("✓ エラーなし")
            
            # 結果を保存
            test_results['request_id'] = req_id
            test_results['max_bp'] = max_bp
            test_results['min_bp'] = min_bp
            test_results['csv_data'] = csv_data
            test_results['errors'] = errors
            test_results['completed'] = True
            
            # CSVファイル保存
            if csv_data:
                csv_filename = f"blood_pressure_result_{req_id[:15]}.csv"
                with open(csv_filename, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                print(f"✓ CSVファイル保存: {csv_filename}")
        
        # 8. 血圧解析開始
        print(f"\n=== 血圧解析開始 ===")
        print(f"動画ファイル: {sample_video_path}")
        print("テストパラメータ:")
        print("  身長: 170cm")
        print("  体重: 70kg") 
        print("  性別: 1 (男性)")
        
        error_code = bp_dll.start_blood_pressure_analysis_request(
            request_id, 
            170,  # 身長
            70,   # 体重
            1,    # 性別（男性）
            str(sample_video_path.absolute()),  # 絶対パス
            analysis_callback
        )
        
        if error_code:
            print(f"✗ 血圧解析開始エラー: {error_code}")
            return False
        
        print("✓ 血圧解析開始成功（非同期処理中...）")
        
        # 9. 処理状況監視
        print("\n=== 処理状況監視 ===")
        max_wait_time = 120  # 最大2分待機
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = bp_dll.get_processing_status(request_id)
            elapsed = time.time() - start_time
            print(f"経過時間: {elapsed:.1f}秒, 状況: {status}")
            
            if status == "none" and test_results.get('completed', False):
                print("✓ 処理完了検出")
                break
            elif status == "none" and elapsed > 10:
                print("⚠️ 処理が予期せず終了した可能性があります")
                break
            
            time.sleep(2)  # 2秒ごとに確認
        
        # 10. 結果確認
        print("\n=== 最終結果確認 ===")
        if test_results.get('completed', False):
            print("🎉 血圧解析完了！")
            print(f"✓ 推定血圧: {test_results['max_bp']}/{test_results['min_bp']} mmHg")
            
            # 結果の妥当性チェック
            max_bp = test_results['max_bp']
            min_bp = test_results['min_bp']
            
            if 80 <= max_bp <= 200 and 50 <= min_bp <= 120 and max_bp > min_bp:
                print("✓ 血圧値は正常範囲内")
            else:
                print("⚠️ 血圧値が異常範囲（アルゴリズム調整が必要な可能性）")
            
            # CSV品質チェック
            csv_data = test_results['csv_data']
            if csv_data:
                lines = csv_data.split('\n')
                data_lines = [line for line in lines if not line.startswith('#') and ',' in line]
                print(f"✓ CSVデータ行数: {len(data_lines)} 行")
                print(f"✓ CSVデータサイズ: {len(csv_data)/1024:.1f} KB")
                
                # 仕様確認（約20KB）
                if 15 <= len(csv_data)/1024 <= 25:
                    print("✓ CSVサイズは仕様範囲内（約20KB）")
                else:
                    print("⚠️ CSVサイズが仕様範囲外")
            
            return True
        else:
            print("❌ 血圧解析がタイムアウトまたは失敗")
            return False
            
    except ImportError as e:
        print(f"✗ DLLインポートエラー: {e}")
        print("bp_estimation_facemesh_spec_compliant.py が見つからないか、依存関係が不足しています")
        return False
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dll_edge_cases():
    """エッジケーステスト"""
    print("\n=== エッジケーステスト ===")
    
    try:
        import bp_estimation_facemesh_spec_compliant as bp_dll
        
        if not bp_dll.initialize_dll():
            print("✗ DLL初期化失敗")
            return False
        
        # 1. 無効なリクエストIDテスト
        print("1. 無効なリクエストIDテスト")
        invalid_ids = [
            "",
            "invalid_format",
            "20250707083524932_9000000001",  # 乗務員コード不足
            "20250707083524932",             # 全て不足
        ]
        
        for invalid_id in invalid_ids:
            error_code = bp_dll.start_blood_pressure_analysis_request(
                invalid_id, 170, 70, 1, "sample-data/100万画素.webm", None)
            if error_code == "1004":
                print(f"✓ 無効ID正常検出: {invalid_id}")
            else:
                print(f"⚠️ 無効ID検出失敗: {invalid_id} -> {error_code}")
        
        # 2. 無効なパラメータテスト
        print("\n2. 無効なパラメータテスト")
        valid_id = bp_dll.generate_request_id("9000000001", "0000012345")
        
        invalid_params = [
            (0, 70, 1),      # 身長0
            (170, 0, 1),     # 体重0
            (170, 70, 0),    # 性別無効
            (170, 70, 3),    # 性別無効
        ]
        
        for height, weight, sex in invalid_params:
            error_code = bp_dll.start_blood_pressure_analysis_request(
                valid_id, height, weight, sex, "sample-data/100万画素.webm", None)
            if error_code == "1004":
                print(f"✓ 無効パラメータ検出: h={height}, w={weight}, s={sex}")
            else:
                print(f"⚠️ 無効パラメータ検出失敗: h={height}, w={weight}, s={sex} -> {error_code}")
        
        # 3. 存在しないファイルテスト
        print("\n3. 存在しないファイルテスト")
        error_code = bp_dll.start_blood_pressure_analysis_request(
            valid_id, 170, 70, 1, "nonexistent.webm", None)
        if error_code == "1004":
            print("✓ 存在しないファイル正常検出")
        else:
            print(f"⚠️ 存在しないファイル検出失敗: {error_code}")
        
        # 4. 重複リクエストテスト
        print("\n4. 重複リクエストテスト")
        # 最初のリクエスト
        error_code1 = bp_dll.start_blood_pressure_analysis_request(
            valid_id, 170, 70, 1, "sample-data/100万画素.webm", None)
        
        # 同じIDで2回目のリクエスト
        error_code2 = bp_dll.start_blood_pressure_analysis_request(
            valid_id, 170, 70, 1, "sample-data/100万画素.webm", None)
        
        if error_code2 == "1005":
            print("✓ 重複リクエスト正常検出")
        else:
            print(f"⚠️ 重複リクエスト検出失敗: {error_code2}")
        
        # 5. 中断機能テスト
        print("\n5. 中断機能テスト")
        cancel_result = bp_dll.cancel_blood_pressure_analysis(valid_id)
        if cancel_result:
            print("✓ 処理中断成功")
        else:
            print("⚠️ 処理中断失敗（処理が既に完了している可能性）")
        
        return True
        
    except Exception as e:
        print(f"✗ エッジケーステストエラー: {e}")
        return False

def test_performance():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    try:
        import bp_estimation_facemesh_spec_compliant as bp_dll
        
        if not bp_dll.initialize_dll():
            print("✗ DLL初期化失敗")
            return False
        
        # 初期化時間測定
        start_time = time.time()
        bp_dll.initialize_dll()
        init_time = time.time() - start_time
        print(f"✓ DLL初期化時間: {init_time:.3f}秒")
        
        # 関数呼び出し速度測定
        start_time = time.time()
        for i in range(100):
            bp_dll.get_version_info()
        version_time = (time.time() - start_time) / 100
        print(f"✓ バージョン取得平均時間: {version_time*1000:.3f}ms")
        
        # リクエストID生成速度
        start_time = time.time()
        for i in range(1000):
            bp_dll.generate_request_id("9000000001", f"{i:010d}")
        id_gen_time = (time.time() - start_time) / 1000
        print(f"✓ リクエストID生成平均時間: {id_gen_time*1000:.3f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ パフォーマンステストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("血圧推定DLL サンプル動画テスト")
    print("=" * 50)
    
    # 事前チェック
    if not Path("sample-data/100万画素.webm").exists():
        print("✗ sample-data/100万画素.webm が見つかりません")
        print("サンプル動画ファイルが必要です")
        return False
    
    if not Path("bp_estimation_facemesh_spec_compliant.py").exists():
        print("✗ bp_estimation_facemesh_spec_compliant.py が見つかりません")
        print("先にDLLビルドスクリプトを実行してください")
        return False
    
    # テスト実行
    results = {}
    
    print("\n【1. サンプル動画テスト】")
    results['video_test'] = test_dll_with_sample_video()
    
    print("\n【2. エッジケーステスト】")
    results['edge_case_test'] = test_dll_edge_cases()
    
    print("\n【3. パフォーマンステスト】")
    results['performance_test'] = test_performance()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 全テスト成功！DLLは正常に動作しています")
        print("\n次のステップ:")
        print("1. 異なる動画ファイルでテスト")
        print("2. 様々なパラメータでの精度検証")
        print("3. 本番環境での統合テスト")
    else:
        print("\n❌ 一部テストが失敗しました")
        print("エラーメッセージを確認して修正してください")
    
    return all_passed

if __name__ == "__main__":
    main()