
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from network import GRconvNet
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results
from skimage.filters import gaussian
import time
import tensorrt as trt
import torch
import torch_tensorrt

if torch.cuda.is_available():
    device = torch.device("cuda")
Dataset = get_dataset("cornell")
test_dataset = Dataset('/home/loahit/Downloads/archive',output_size=224,ds_rotate=True,random_rotate=True,random_zoom=True,include_depth=True,include_rgb=True)

indices = list(range(test_dataset.length))
split_=0.9
ds_shuffle=False
random_seed=123
num_workers=1
n_grasps=1
iou_threshold=0.25
iou_eval=True
vis=False
split = int(np.floor(split_ * test_dataset.length))
if ds_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
val_indices = indices[split:]
val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
testing_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )
net=GRconvNet()
net = torch.load("/home/loahit/GRconvnet/epoch_42_iou_0.92")

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda"),
)

trt_mod = torch_tensorrt.compile(net, inputs=[torch_tensorrt.Input((1, 4, 224, 224))],
                                    enabled_precisions={torch.float, torch.half, torch.int8},
                                    calibrator=calibrator,
                                    device={
                                         "device_type": torch_tensorrt.DeviceType.GPU,
                                         "gpu_id": 0,
                                         "dla_core": 0,
                                         "allow_gpu_fallback": False,
                                         "disable_tf32": False
                                     })
