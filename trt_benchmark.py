import tensorrt as trt
import torch
import torch_tensorrt
import time
import numpy as np
from network import GRconvNet

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


device = torch.device("cuda")

#device = torch.device("cpu")

model = torch.load("/home/loahit/GRconvnet/epoch_42_iou_0.92")
model= model.to(device)

model_gpu=model.to("cuda")
input_data = torch.empty([1, 4, 224, 224]).to(device)
traced_model = torch.jit.trace(model_gpu, input_data)

trt_model_fp16 = torch_tensorrt.compile(
    traced_model,
    inputs = [torch_tensorrt.Input((1, 4, 224, 224), dtype=torch.float32)],
    enabled_precisions = {torch.float16}
)

trt_model_fp32 = torch_tensorrt.compile(
    traced_model,
    inputs = [torch_tensorrt.Input((1, 4, 224, 224), dtype=torch.float32)],
    enabled_precisions = {torch.float32}
)


def benchmark(model, device="cuda", input_shape=(1, 4, 224, 224), dtype='fp32', nwarmup=50, nruns=50):
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average Grasp Inference time: %.2f ms'%(np.mean(timings)*1000))

#benchmark(model, device="cpu") for cpu
print("FP16 precision:")
benchmark(trt_model_fp16,dtype='fp16')
print("FP32 precision:")
benchmark(trt_model_fp32,dtype='fp32')  

torch.jit.save(trt_model_fp16, "trt_model_fp16.jit.pt")
torch.jit.save(trt_model_fp32, "trt_model_fp16.jit.pt")