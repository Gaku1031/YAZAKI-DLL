================================================================================
                    血圧推定DLL 使用説明書 v1.0
================================================================================

■ 概要
本DLLは、30秒間のWebM動画ファイルから非侵襲的に血圧を推定するライブラリです。
顔の映像解析により、最高血圧と最低血圧を推定し、検証用のPPGデータをCSV形式で出力します。

■ システム要件
- OS: Windows 10/11 (64bit推奨)
- メモリ: 4GB以上推奨
- ディスク容量: 100MB以上の空き容量
- Pythonランタイム: 不要（DLL内に内包済み）

■ ファイル構成
BloodPressureEstimation.dll - メインDLLファイル
models/ - 血圧推定モデルファイル（任意）
README.txt - 本説明書

■ 対応動画仕様
- 撮影時間: 30秒
- フレームレート: 30FPS
- 解像度: 1280x720（100万画素）
- コンテナ: WebM
- コーデック: VP8
- 拡張子: .webm
- ビットレート: 約2.5Mbps

■ DLL関数一覧

【1】InitializeDLL(model_dir)
機能: DLLを初期化します
引数: model_dir - モデルディレクトリパス（省略可）
戻り値: bool - 成功時True、失敗時False

【2】StartBloodPressureAnalysisRequest(request_id, height, weight, sex, movie_path, callback)
機能: 血圧解析を開始します（非同期処理）
引数:
  - request_id: リクエストID（形式: yyyyMMddHHmmssfff_顧客コード_乗務員コード）
  - height: 身長（cm、100-250の範囲）
  - weight: 体重（kg、30-200の範囲）
  - sex: 性別（1=男性、2=女性）
  - movie_path: WebM動画ファイルのフルパス
  - callback: 結果通知用コールバック関数
戻り値: 文字列 - エラーコード（正常時は空文字）

【3】GetProcessingStatus(request_id)
機能: 指定リクエストの処理状況を取得します
引数: request_id - 対象リクエストID
戻り値: 文字列 - "none"（未処理）または "processing"（処理中）

【4】CancelBloodPressureAnalysis(request_id)
機能: 指定リクエストの処理を中断します
引数: request_id - 対象リクエストID
戻り値: bool - 中断成功時True、失敗時False

【5】GetVersionInfo()
機能: DLLのバージョン情報を取得します
引数: なし
戻り値: 文字列 - バージョン情報（例: "v1.0.0"）

■ コールバック関数仕様

コールバック関数は以下の形式で実装してください：

void Callback(char* request_id, int max_bp, int min_bp, char* csv_data, void* errors)

引数:
- request_id: リクエスト時と同一のID
- max_bp: 最高血圧（mmHg、0-999の範囲）
- min_bp: 最低血圧（mmHg、0-999の範囲）
- csv_data: PPGローデータ（CSV形式、約20KB）
- errors: エラー情報（エラー時のみ）

■ エラーコード一覧

1001: DLLが初期化されていない
1002: デバイス接続失敗（本DLLでは通常発生しません）
1003: キャリブレーション未完了（本DLLでは通常発生しません）
1004: 入力パラメータ不正
1005: 測定中リクエスト受付不可
1006: DLL内部処理エラー

■ 使用手順

1. InitializeDLL()でDLLを初期化
2. リクエストIDを生成（形式: yyyyMMddHHmmssfff_顧客コード_乗務員コード）
3. StartBloodPressureAnalysisRequest()で解析開始
4. コールバックで結果を受信
5. 必要に応じてGetProcessingStatus()で状況確認

■ リクエストID形式

${yyyyMMddHHmmssfff}_${顧客コード}_${乗務員コード}

例: 20250707083524932_9000000001_0000012345

- yyyyMMddHHmmssfff: タイムスタンプ（17桁）
- 顧客コード: 10桁の数字
- 乗務員コード: 10桁の数字

■ 出力CSVデータ形式

Time(s),rPPG_Signal,Peak_Flag,Heart_Rate(bpm),Signal_Quality
0.000,0.234567,0,0,75
0.067,0.245678,0,0,78
...

列説明:
- Time(s): 時間（秒）
- rPPG_Signal: 脈波信号値
- Peak_Flag: ピーク検出フラグ（1=ピーク、0=非ピーク）
- Heart_Rate(bpm): 心拍数（10秒窓での平均）
- Signal_Quality: 信号品質（0-100）

■ 注意事項

1. 動画ファイルは指定仕様に準拠してください
2. 顔が明瞭に映っている動画を使用してください
3. 照明条件の良い環境で撮影された動画を推奨します
4. 同一リクエストIDでの重複実行はエラーになります
5. 処理時間は動画の品質により10-60秒程度かかります

■ トラブルシューティング

【問題】初期化に失敗する
【対策】modelsフォルダが存在することを確認してください

【問題】解析が開始されない（エラーコード1004）
【対策】リクエストID形式、パラメータ範囲を確認してください

【問題】解析結果が不正確
【対策】動画品質、照明条件、顔の映り方を確認してください

【問題】処理が完了しない
【対策】CancelBloodPressureAnalysis()で中断後、再実行してください

■ 技術サポート

本DLLに関するお問い合わせは、契約に基づく技術サポート窓口までご連絡ください。

■ 免責事項

- 本DLLは研究・開発目的での使用を想定しています
- 医療診断には使用しないでください
- 推定結果の精度は撮影条件により変動します
- 本DLLの使用により生じた損害について、開発元は一切の責任を負いません

================================================================================
Copyright (C) 2025 IKI Japan/Yazaki. All rights reserved.
================================================================================
