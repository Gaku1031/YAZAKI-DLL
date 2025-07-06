# 動画からRRIを取得し、そのデータを使って血圧を推定

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# === 事前に学習済みモデルを保存（1回だけ実行）===
def train_and_save_models(features, sbp_labels, dbp_labels, model_dir):
    model_sbp = RandomForestRegressor(random_state=42)
    model_dbp = RandomForestRegressor(random_state=42)
    model_sbp.fit(features, sbp_labels)
    model_dbp.fit(features, dbp_labels)
    joblib.dump(model_sbp, f"{model_dir}/model_sbp.pkl")
    joblib.dump(model_dbp, f"{model_dir}/model_dbp.pkl")

# === RRIから特徴量抽出・予測実行 ===
def predict_bp_from_rri(csv_path, height_cm, weight_kg, sex_male_1_female_0, model_dir):
    df = pd.read_csv(csv_path)
    if 'RRI(s)' not in df.columns:
        raise ValueError("CSVに 'RRI(s)' カラムが存在しません。")
    
    # rri = df['RRI(s)'].copy()
    # rri[(rri < 0.4) | (rri > 1.2)] = np.nan
    # rri = rri.ffill()
    
    # if rri.isna().any():
    #     raise ValueError("RRIに有効な値が含まれていません。")

    rri = df['RRI(s)'].copy()
    rri[(rri < 0.4) | (rri > 1.2)] = np.nan
    rri = rri.ffill().bfill()

    if rri.isna().any():
        raise ValueError("RRIに有効な値が含まれていません。")
    
    bmi = weight_kg / ((height_cm / 100) ** 2)
    
    feature_vector = np.array([
        rri.mean(),
        rri.std(),
        rri.min(),
        rri.max(),
        bmi,
        sex_male_1_female_0
    ]).reshape(1, -1)
    
    # 学習済みモデルの読み込み
    model_sbp = joblib.load(f"{model_dir}/model_sbp.pkl")
    model_dbp = joblib.load(f"{model_dir}/model_dbp.pkl")
    
    pred_sbp = model_sbp.predict(feature_vector)[0]
    pred_dbp = model_dbp.predict(feature_vector)[0]
    
    return pred_sbp, pred_dbp

height = 170.0  # cm
weight = 65.0   # kg
sex = 1       # 男性=1, 女性=0
csv_path = "/Users/gakuinoue/workspace/IKI/BP-estimation/result/2025-07-02/rppg_data_20250702_232429_30万画素.csv"
model_dir = "/Users/gakuinoue/workspace/IKI/BP-estimation/2025-06-29/result/models"  # モデル保存ディレクトリ（要事前学習）

# 予測の実行
pred_sbp, pred_dbp = predict_bp_from_rri(csv_path, height, weight, sex, model_dir)
print(f"予測収縮期血圧（SBP）: {pred_sbp:.2f} mmHg")
print(f"予測拡張期血圧（DBP）: {pred_dbp:.2f} mmHg")
