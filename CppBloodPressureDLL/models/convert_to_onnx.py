#!/usr/bin/env python3
"""
Convert scikit-learn models to ONNX format for C++ usage
"""

import os
import sys
import joblib
import numpy as np
import onnx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def create_sample_models():
    """Create sample models if original models don't exist"""
    print("Creating sample models...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: [rri_mean, rri_std, rri_min, rri_max, bmi, sex]
    rri_mean = np.random.normal(0.8, 0.1, n_samples)
    rri_std = np.random.normal(0.05, 0.02, n_samples)
    rri_min = rri_mean - np.random.uniform(0.1, 0.3, n_samples)
    rri_max = rri_mean + np.random.uniform(0.1, 0.3, n_samples)
    bmi = np.random.normal(23, 3, n_samples)
    sex = np.random.choice([0, 1], n_samples)
    
    X = np.column_stack([rri_mean, rri_std, rri_min, rri_max, bmi, sex])
    
    # Generate realistic blood pressure values
    # Simple linear relationships for demonstration
    sbp_base = 120 + (bmi - 23) * 2 + sex * 10 + np.random.normal(0, 10, n_samples)
    sbp_rri_effect = (rri_mean - 0.8) * 50  # RRI effect
    y_sbp = sbp_base + sbp_rri_effect
    y_sbp = np.clip(y_sbp, 90, 200)  # Realistic range
    
    dbp_base = 80 + (bmi - 23) * 1.5 + sex * 5 + np.random.normal(0, 8, n_samples)
    dbp_rri_effect = (rri_mean - 0.8) * 30  # RRI effect
    y_dbp = dbp_base + dbp_rri_effect
    y_dbp = np.clip(y_dbp, 60, 120)  # Realistic range
    
    # Train models
    X_train, X_test, y_sbp_train, y_sbp_test = train_test_split(X, y_sbp, test_size=0.2, random_state=42)
    _, _, y_dbp_train, y_dbp_test = train_test_split(X, y_dbp, test_size=0.2, random_state=42)
    
    # SBP model
    model_sbp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_sbp.fit(X_train, y_sbp_train)
    
    # DBP model
    model_dbp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_dbp.fit(X_train, y_dbp_train)
    
    # Evaluate models
    sbp_pred = model_sbp.predict(X_test)
    dbp_pred = model_dbp.predict(X_test)
    
    print(f"SBP Model - R²: {r2_score(y_sbp_test, sbp_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_sbp_test, sbp_pred)):.3f}")
    print(f"DBP Model - R²: {r2_score(y_dbp_test, dbp_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_dbp_test, dbp_pred)):.3f}")
    
    # Save models
    joblib.dump(model_sbp, "model_sbp.pkl")
    joblib.dump(model_dbp, "model_dbp.pkl")
    
    print("Sample models created and saved")
    return model_sbp, model_dbp

def convert_sklearn_to_onnx(model_path, output_path, model_name):
    """Convert a scikit-learn model to ONNX format"""
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"Loaded model: {model_path}")
        
        # Define input type - 6 features (rri_mean, rri_std, rri_min, rri_max, bmi, sex)
        initial_type = [('float_input', FloatTensorType([None, 6]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save the ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Successfully converted {model_name} to ONNX: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error converting {model_name}: {str(e)}")
        return False

def main():
    """Main conversion function"""
    print("Starting model conversion to ONNX format...")
    
    # Check if original models exist (try both local and parent directory)
    sbp_model_path = "model_sbp.pkl"
    dbp_model_path = "model_dbp.pkl"
    
    # Try to find models in parent directory first
    if not os.path.exists(sbp_model_path) and os.path.exists("../model_sbp.pkl"):
        sbp_model_path = "../model_sbp.pkl"
    if not os.path.exists(dbp_model_path) and os.path.exists("../model_dbp.pkl"):
        dbp_model_path = "../model_dbp.pkl"
    
    # If models don't exist, create sample models
    if not os.path.exists(sbp_model_path) or not os.path.exists(dbp_model_path):
        print("Original models not found. Creating sample models...")
        create_sample_models()
    
    # Convert models
    success_count = 0
    
    # Convert SBP model
    if convert_sklearn_to_onnx(sbp_model_path, "model_sbp.onnx", "SBP"):
        success_count += 1
    
    # Convert DBP model
    if convert_sklearn_to_onnx(dbp_model_path, "model_dbp.onnx", "DBP"):
        success_count += 1
    
    print(f"\nConversion completed: {success_count}/2 models converted successfully")
    
    if success_count == 2:
        print("All models converted successfully!")
        print("Files created:")
        print("- model_sbp.onnx")
        print("- model_dbp.onnx")
        return 0
    else:
        print("Some models failed to convert. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
