import onnx

# モデルファイルのパスを指定
model = onnx.load("/Users/gakuinoue/workspace/IKI/YAZAKI-DLL/CppBloodPressureDLL/models/systolicbloodpressure.onnx")

# 入力名を表示
for input in model.graph.input:
    print("Input name:", input.name)
