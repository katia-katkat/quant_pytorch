import torch
from models.common import DetectMultiBackend
import torch.quantization

model_q = DetectMultiBackend(weights='yolov5s.pt')
""" for n, p in model.named_parameters():
    print(n, ": ", p.dtype) """

""" model_q = model.half()
model_q1 = model_q.half()
for n, p in model_q1.named_parameters():
    print(n, ": ", p.dtype) """
#model_q = model.half()
quant_model = torch.quantization.quantize_dynamic(model_q,  dtype=torch.qint8)
for n, p in quant_model.named_parameters():
    print(n, ": ", p.dtype)
#quant_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype= torch.qint8)

