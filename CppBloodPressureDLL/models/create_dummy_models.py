#!/usr/bin/env python3
"""
Create dummy ONNX models for blood pressure estimation
These are placeholder models for testing purposes
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper
import os


def create_dummy_model(model_name, input_shape=(1, 3, 224, 224), output_shape=(1, 1)):
    """Create a dummy ONNX model for testing"""

    # Create input
    input_name = "input"
    input_type = onnx.TensorProto.FLOAT
    input_shape = list(input_shape)

    # Create output
    output_name = "output"
    output_type = onnx.TensorProto.FLOAT
    output_shape = list(output_shape)

    # Create a simple model: input -> reshape -> output
    nodes = [
        helper.make_node(
            "Identity",
            inputs=[input_name],
            outputs=[output_name],
            name="identity_node"
        )
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes,
        model_name,
        [helper.make_tensor_value_info(input_name, input_type, input_shape)],
        [helper.make_tensor_value_info(output_name, output_type, output_shape)]
    )

    # Create the model
    model = helper.make_model(graph, producer_name="BloodPressureDLL")

    return model


def main():
    """Create dummy SBP and DBP models"""

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Create SBP model
    print("Creating dummy SBP model...")
    sbp_model = create_dummy_model("SystolicBloodPressure")
    onnx.save(sbp_model, "models/model_sbp.onnx")
    print("SBP model saved: models/model_sbp.onnx")

    # Create DBP model
    print("Creating dummy DBP model...")
    dbp_model = create_dummy_model("DiastolicBloodPressure")
    onnx.save(dbp_model, "models/model_dbp.onnx")
    print("DBP model saved: models/model_dbp.onnx")

    # Create dummy OpenCV DNN files
    print("Creating dummy OpenCV DNN files...")

    # Create dummy .pb file (just a text file for testing)
    with open("models/opencv_face_detector_uint8.pb", "w") as f:
        f.write("Dummy OpenCV DNN model file for testing\n")
    print("OpenCV DNN model saved: models/opencv_face_detector_uint8.pb")

    # Create dummy .pbtxt file
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
    print("OpenCV DNN config saved: models/opencv_face_detector.pbtxt")

    print("All dummy model files created successfully!")


if __name__ == "__main__":
    main()
