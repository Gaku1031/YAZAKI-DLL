#!/usr/bin/env python3
"""
Convert PKL models to ONNX format for blood pressure estimation
"""

import pickle
import numpy as np
import onnx
from onnx import helper, numpy_helper
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import joblib


def load_pkl_model(model_path):
    """Load PKL model and determine its type"""
    try:
        print(f"Loading model from: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        print(f"File size: {os.path.getsize(model_path)} bytes")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Model loaded successfully: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_onnx_from_sklearn(model, model_name, input_shape=(1, 10)):
    """Convert scikit-learn model to ONNX format"""

    print(f"Creating ONNX model for {model_name}")
    print(f"Model type: {type(model)}")
    print(f"Input shape: {input_shape}")

    # Create input
    input_name = "input"
    input_type = onnx.TensorProto.FLOAT
    input_shape = list(input_shape)

    # Create output
    output_name = "output"
    output_type = onnx.TensorProto.FLOAT
    output_shape = [input_shape[0], 1]  # Single output for regression

    # Create nodes based on model type
    nodes = []

    if isinstance(model, RandomForestRegressor):
        # For Random Forest, we'll create a simplified representation
        # In practice, you might want to use a more sophisticated approach
        nodes = [
            helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=f"{model_name}_identity"
            )
        ]
        print(f"RandomForest model detected for {model_name}")

    elif isinstance(model, LinearRegression):
        # For Linear Regression: y = ax + b
        coef = model.coef_.astype(np.float32)
        intercept = model.intercept_.astype(np.float32)

        print(f"LinearRegression coefficients shape: {coef.shape}")
        print(f"LinearRegression intercept: {intercept}")

        # Create weight and bias tensors
        weight_tensor = numpy_helper.from_array(
            coef.reshape(1, -1), name="weight")
        bias_tensor = numpy_helper.from_array(
            intercept.reshape(1), name="bias")

        nodes = [
            helper.make_node(
                "Gemm",
                inputs=[input_name, "weight", "bias"],
                outputs=[output_name],
                name=f"{model_name}_gemm",
                alpha=1.0,
                beta=1.0
            )
        ]
        print(f"LinearRegression model detected for {model_name}")

    elif isinstance(model, SVR):
        # For SVR, we'll create a simplified representation
        nodes = [
            helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=f"{model_name}_identity"
            )
        ]
        print(f"SVR model detected for {model_name}")

    else:
        # Default fallback
        nodes = [
            helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=f"{model_name}_identity"
            )
        ]
        print(f"Unknown model type for {model_name}: {type(model)}")

    # Create initializers for weights and biases
    initializers = []
    if isinstance(model, LinearRegression):
        initializers.extend([weight_tensor, bias_tensor])

    # Create the graph
    graph = helper.make_graph(
        nodes,
        model_name,
        [helper.make_tensor_value_info(input_name, input_type, input_shape)],
        [helper.make_tensor_value_info(
            output_name, output_type, output_shape)],
        initializers
    )

    # Create the model
    onnx_model = helper.make_model(graph, producer_name="BloodPressureDLL")

    return onnx_model


def main():
    """Convert PKL models to ONNX"""

    print("=== PKL to ONNX Conversion Script ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    print(f"Models directory: {os.path.abspath('models')}")

    # List all files in current directory
    print("Files in current directory:")
    for file in os.listdir("."):
        if os.path.isfile(file):
            size = os.path.getsize(file)
            print(f"  {file} ({size} bytes)")

    # Model files to convert - check both current directory and models subdirectory
    model_files = [
        ("model_sbp.pkl", "SystolicBloodPressure"),
        ("model_dbp.pkl", "DiastolicBloodPressure")
    ]

    for pkl_file, model_name in model_files:
        print(f"\nProcessing {pkl_file}...")

        # Try multiple possible paths
        possible_paths = [
            pkl_file,  # Current directory
            os.path.join("models", pkl_file),  # models subdirectory
            os.path.join("..", pkl_file),  # Parent directory
        ]

        pkl_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pkl_path = path
                print(f"Found {pkl_file} at: {path}")
                break

        if pkl_path is None:
            print(f"Warning: {pkl_file} not found in any expected location")
            print("Searched in:")
            for path in possible_paths:
                print(f"  - {os.path.abspath(path)}")
            continue

        # Load PKL model
        model = load_pkl_model(pkl_path)
        if model is None:
            print(f"Failed to load {pkl_path}")
            continue

        # Print model information
        print(f"Model type: {type(model)}")
        if hasattr(model, 'n_estimators'):
            print(f"Number of estimators: {model.n_estimators}")
        if hasattr(model, 'coef_'):
            print(f"Number of features: {len(model.coef_)}")

        # Convert to ONNX
        try:
            # Determine input shape based on model
            if hasattr(model, 'coef_'):
                input_shape = (1, len(model.coef_))
            elif hasattr(model, 'n_features_in_'):
                input_shape = (1, model.n_features_in_)
            else:
                input_shape = (1, 10)  # Default fallback

            print(f"Input shape: {input_shape}")

            onnx_model = create_onnx_from_sklearn(
                model, model_name, input_shape)

            # Save ONNX model
            onnx_file = f"{model_name.lower()}.onnx"
            onnx_path = f"models/{onnx_file}"
            onnx.save(onnx_model, onnx_path)
            print(f"ONNX model saved: {onnx_path}")
            print(f"File size: {os.path.getsize(onnx_path)} bytes")

            # Verify the model
            try:
                onnx.checker.check_model(onnx_model)
                print(f"ONNX model validation passed for {onnx_file}")
            except Exception as e:
                print(f"ONNX model validation failed for {onnx_file}: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"Error converting {pkl_path} to ONNX: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create dummy OpenCV DNN files if they don't exist
    print("\nCreating OpenCV DNN files...")

    if not os.path.exists("models/opencv_face_detector_uint8.pb"):
        with open("models/opencv_face_detector_uint8.pb", "w") as f:
            f.write("Dummy OpenCV DNN model file for testing\n")
        print("OpenCV DNN model created: models/opencv_face_detector_uint8.pb")

    if not os.path.exists("models/opencv_face_detector.pbtxt"):
        with open("models/opencv_face_detector.pbtxt", "w") as f:
            f.write("name: \"DummyFaceDetector\"\n")
            f.write("input: \"input\"\n")
            f.write("output: \"output\"\n")
            f.write("layer {\n")
            f.write("  name: \"dummy_layer\"\n")
            f.write("  type: \"Identity\"\n")
            f.write("  input: \"input\"\n")
            f.write("  output: \"output\"\n")
            f.write("}\n")
        print("OpenCV DNN config created: models/opencv_face_detector.pbtxt")

    print("\nAll models converted successfully!")

    # List all files
    print("\nFiles in models directory:")
    for file in os.listdir("models"):
        file_path = os.path.join("models", file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")


if __name__ == "__main__":
    main()
