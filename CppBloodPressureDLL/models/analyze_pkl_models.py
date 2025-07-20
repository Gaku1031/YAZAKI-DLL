#!/usr/bin/env python3
"""
Analyze PKL models to understand their structure and type
"""

import pickle
import numpy as np
import os
import sys


def analyze_pkl_model(model_path):
    """Analyze a PKL model and print its details"""
    print(f"\n=== Analyzing {model_path} ===")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Model type: {type(model)}")
        print(f"Model class: {model.__class__.__name__}")

        # Print model attributes
        print("\nModel attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and not callable(getattr(model, attr)):
                try:
                    value = getattr(model, attr)
                    if isinstance(value, (int, float, str, bool)):
                        print(f"  {attr}: {value}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {attr}: numpy array with shape {value.shape}")
                    elif hasattr(value, '__len__'):
                        print(
                            f"  {attr}: {type(value).__name__} with length {len(value)}")
                    else:
                        print(f"  {attr}: {type(value).__name__}")
                except Exception as e:
                    print(f"  {attr}: Error reading - {e}")

        # Specific analysis for common model types
        if hasattr(model, 'n_estimators'):
            print(f"\nRandomForest details:")
            print(f"  Number of estimators: {model.n_estimators}")
            print(f"  Max depth: {model.max_depth}")
            print(f"  Min samples split: {model.min_samples_split}")

        if hasattr(model, 'coef_'):
            print(f"\nLinear model details:")
            print(f"  Number of features: {len(model.coef_)}")
            print(f"  Intercept: {model.intercept_}")
            print(f"  Coefficients shape: {model.coef_.shape}")

        if hasattr(model, 'support_vectors_'):
            print(f"\nSVR details:")
            print(
                f"  Number of support vectors: {len(model.support_vectors_)}")
            print(f"  Support vectors shape: {model.support_vectors_.shape}")

        if hasattr(model, 'n_features_in_'):
            print(f"\nFeature information:")
            print(f"  Number of input features: {model.n_features_in_}")

        if hasattr(model, 'feature_names_in_'):
            print(f"  Feature names: {list(model.feature_names_in_)}")

        # Test prediction if possible
        print(f"\nTesting prediction capability:")
        try:
            # Create dummy input based on model features
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif hasattr(model, 'coef_'):
                n_features = len(model.coef_)
            else:
                n_features = 10  # Default fallback

            dummy_input = np.random.random((1, n_features))
            prediction = model.predict(dummy_input)
            print(f"  Prediction successful: {prediction}")
            print(f"  Prediction shape: {prediction.shape}")
            print(f"  Prediction type: {type(prediction)}")
        except Exception as e:
            print(f"  Prediction failed: {e}")

        return model

    except Exception as e:
        print(f"Error analyzing {model_path}: {e}")
        return None


def main():
    """Analyze all PKL models"""

    model_files = ["model_sbp.pkl", "model_dbp.pkl"]

    for model_file in model_files:
        if os.path.exists(model_file):
            model = analyze_pkl_model(model_file)
        else:
            print(f"\nWarning: {model_file} not found")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
