#!/usr/bin/env python3
"""
Convert PKL models to ONNX format for blood pressure estimation
"""

import os
import sys
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def convert_model(pkl_path, onnx_path):
    # モデルロード
    try:
        model = joblib.load(pkl_path)
        print(f"Loaded model from {pkl_path}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # 入力次元数の推定
    if hasattr(model, 'n_features_in_'):
        input_dim = model.n_features_in_
    elif hasattr(model, 'coef_'):
        input_dim = len(model.coef_)
    else:
        print("ERROR: モデルから入力次元数を取得できません")
        sys.exit(1)

    # skl2onnxで変換
    try:
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"ERROR: skl2onnx変換失敗: {e}")
        sys.exit(1)

    # ファイルサイズチェック
    size = os.path.getsize(onnx_path)
    print(f"ONNX file size: {size} bytes")
    if size < 100 * 1024:
        print("ERROR: ONNXファイルが小さすぎます（変換失敗の可能性）")
        sys.exit(1)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    convert_model("/Users/gakuinoue/workspace/IKI/YAZAKI-DLL/models/model_sbp.pkl", "/Users/gakuinoue/workspace/IKI/YAZAKI-DLL/models/systolicbloodpressure.onnx")
    convert_model("/Users/gakuinoue/workspace/IKI/YAZAKI-DLL/models/model_dbp.pkl", "/Users/gakuinoue/workspace/IKI/YAZAKI-DLL/models/diastolicbloodpressure.onnx")
