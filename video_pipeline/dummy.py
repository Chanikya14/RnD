import onnxruntime as ort

session = ort.InferenceSession("yolov5s.onnx", providers=['CUDAExecutionProvider'])
for inp in session.get_inputs():
    print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
