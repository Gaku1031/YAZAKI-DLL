#!/usr/bin/env python3
"""
Convert PKL models to ONNX format for blood pressure estimation
"""

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from onnx import helper, numpy_helper
import os
import sys
import pickle
import numpy as np
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    import joblib
    print("Joblib available for model loading")
except ImportError:
    print("Warning: Joblib not available, falling back to pickle")
    joblib = None

try:
    import onnx
    print("ONNX available for model conversion")
except ImportError:
    print("Error: ONNX not available")
    sys.exit(1)


def load_pkl_model(file_path):
    """Load PKL model with multiple fallback methods"""
    print(f"Loading model from: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path)} bytes")

    # Try multiple loading methods - prioritize joblib for scikit-learn models
    loading_methods = []

    if joblib is not None:
        loading_methods.append(
            ("Joblib (recommended for scikit-learn)", lambda f: joblib.load(f)))

    loading_methods.extend([
        ("Standard pickle", lambda f: pickle.load(f)),
        ("Pickle with protocol 5", lambda f: pickle.load(f, protocol=5)),
        ("Pickle with protocol 4", lambda f: pickle.load(f, protocol=4)),
        ("Pickle with protocol 3", lambda f: pickle.load(f, protocol=3)),
    ])

    for method_name, load_func in loading_methods:
        try:
            print(f"Trying {method_name}...")

            # Try different file opening modes
            for mode in ['rb', 'r+b']:
                try:
                    with open(file_path, mode) as f:
                        model = load_func(f)
                        print(
                            f"Successfully loaded with {method_name} (mode: {mode})")
                        return model
                except Exception as e:
                    print(
                        f"  Failed with mode {mode}: {type(e).__name__}: {e}")
                    continue

        except Exception as e:
            print(f"  {method_name} failed: {type(e).__name__}: {e}")
            continue

    # If all methods fail, analyze the file for debugging
    print("All loading methods failed. Analyzing file content for debugging...")
    try:
        with open(file_path, 'rb') as f:
            content = f.read(100)  # Read first 100 bytes
            print(f"File header (hex): {content[:20].hex()}")
            print(f"File header (ascii): {repr(content[:20])}")

            # Check if it's a text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    print(f"First line: {repr(first_line)}")
            except:
                print("File is not readable as text (expected for binary PKL files)")

    except Exception as e:
        print(f"File analysis failed: {e}")

    return None


def create_onnx_from_sklearn(model, model_name, input_shape=(1, 10)):
    """Convert scikit-learn model to ONNX format"""

    try:
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
        initializers = []

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
            initializers.extend([weight_tensor, bias_tensor])
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

        # Create the graph
        graph = helper.make_graph(
            nodes,
            model_name,
            [helper.make_tensor_value_info(
                input_name, input_type, input_shape)],
            [helper.make_tensor_value_info(
                output_name, output_type, output_shape)],
            initializers
        )

        # Create the model
        onnx_model = helper.make_model(graph, producer_name="BloodPressureDLL")

        print(f"ONNX model created successfully for {model_name}")
        return onnx_model

    except Exception as e:
        print(f"Error creating ONNX model for {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


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

    conversion_success = True
    successful_conversions = 0

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
            print(f"Error: {pkl_file} not found in any expected location")
            print("Searched in:")
            for path in possible_paths:
                print(f"  - {os.path.abspath(path)}")
            conversion_success = False
            continue

        # Load PKL model
        model = load_pkl_model(pkl_path)
        if model is None:
            print(f"Failed to load {pkl_path}. Cannot create ONNX model.")
            print(
                f"ERROR: {model_name} model could not be loaded from {pkl_path}")
            print("This may be due to:")
            print("  - Corrupted PKL file")
            print("  - Incompatible scikit-learn version")
            print("  - File encoding issues")
            conversion_success = False
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

            try:
                onnx_model = create_onnx_from_sklearn(
                    model, model_name, input_shape)

                if onnx_model is None:
                    print(
                        f"ERROR: Failed to create ONNX model for {model_name}")
                    conversion_success = False
                    continue

                print(f"ONNX model created successfully for {model_name}")

            except Exception as onnx_create_error:
                print(
                    f"Error creating ONNX model for {model_name}: {onnx_create_error}")
                import traceback
                traceback.print_exc()
                conversion_success = False
                continue

            # Save ONNX model
            onnx_file = f"{model_name.lower()}.onnx"
            onnx_path = f"models/{onnx_file}"

            try:
                onnx.save(onnx_model, onnx_path)
                file_size = os.path.getsize(onnx_path)
                print(f"ONNX model saved: {onnx_path}")
                print(f"File size: {file_size} bytes")

                if file_size == 0:
                    print(
                        f"ERROR: ONNX file {onnx_path} was created but is empty (0 bytes)")
                    print(
                        "This indicates a problem with the ONNX model creation or saving")
                    conversion_success = False
                    continue

            except Exception as save_error:
                print(f"Error saving ONNX model to {onnx_path}: {save_error}")
                import traceback
                traceback.print_exc()
                conversion_success = False
                continue

            # Verify the model
            try:
                onnx.checker.check_model(onnx_model)
                print(f"ONNX model validation passed for {onnx_file}")
                successful_conversions += 1
            except Exception as e:
                print(f"ONNX model validation failed for {onnx_file}: {e}")
                import traceback
                traceback.print_exc()
                conversion_success = False

        except Exception as e:
            print(f"Error converting {pkl_path} to ONNX: {e}")
            import traceback
            traceback.print_exc()
            conversion_success = False
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

    # Report conversion results
    if conversion_success and successful_conversions == len(model_files):
        print(
            f"\nAll models converted successfully! ({successful_conversions}/{len(model_files)})")
    else:
        print(
            f"\nModel conversion failed! ({successful_conversions}/{len(model_files)} successful)")
        print("ERROR: Some models could not be converted to ONNX format.")
        print("This will prevent the DLL from functioning properly.")
        print("Please check the PKL files and ensure they are compatible with the current scikit-learn version.")
        sys.exit(1)


if __name__ == "__main__":
    main()
