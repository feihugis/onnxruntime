from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import load_model, save_model

onnx_model = load_model('/model/data/onnx-opt/model_onnx_debug_base_fp16.onnx')
new_onnx_model = convert_float_to_float16(onnx_model, keep_io_types=True)
save_model(new_onnx_model, '/model/data/onnx-opt/model_onnx_debug_base_fp16_1.onnx')
