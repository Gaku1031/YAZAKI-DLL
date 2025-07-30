# FFmpeg H.264 対応版 クイックアップデート

## 問題

現在の ffmpeg.exe は h264 デコーダーをサポートしていないため、h264 エンコードされた webm ファイルでエラーが発生します。

## 解決手順

### 1. GitHub Actions を実行

1. GitHub リポジトリの「Actions」タブに移動
2. 「Build FFmpeg with H.264 Support」を選択
3. 「Run workflow」をクリック
4. ビルド完了まで待機（約 10-15 分）

### 2. Artifacts をダウンロード

1. ビルド完了後、「ffmpeg-ultra-slim」をダウンロード
2. プロジェクトルートに展開

### 3. ffmpeg.exe を更新

```bash
update_ffmpeg.bat
```

### 4. 動作確認

```bash
test_ffmpeg_h264.bat
```

## 変更点

- `--enable-decoder='h264,vp9'` を追加
- `--enable-parser='h264,vp9'` を追加
- 静的リンクでビルド

## 期待される結果

- h264 エンコードされた webm ファイルが正常に読み込める
- エラー「Decoding requested, but no decoder found for: h264」が解決
