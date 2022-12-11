import json
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
import matplotlib.pyplot as plt

from utils.dataset_processing.grasp import detect_grasps

device = torch.device("cuda")
def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    plt.pause(0.1)
    fig.canvas.draw()
    cv2.waitKey(0)



def post_process_output(q_img, cos_img, sin_img, width_img):

    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def test():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    Dataset = get_dataset("cornell")
    test_dataset = Dataset('/home/loahit/Downloads/archive',output_size=224,ds_rotate=True,random_rotate=True,random_zoom=True,include_depth=True,include_rgb=True)

    indices = list(range(test_dataset.length))
    split_=0.9
    ds_shuffle=False
    random_seed=123
    num_workers=8
    n_grasps=2
    iou_threshold=0.25
    iou_eval=True
    vis=False
    split = int(np.floor(split_ * test_dataset.length))
    if ds_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )
    
    
    net = torch.load("/home/loahit/GRconvnet/trained _Models/epoch_45_iou_0.98")
    #net = torch.load("/home/loahit/GRconvnet/trained _Models/trt_model_fp16.jit.pt")
    #net = torch.load("/home/loahit/GRconvnet/trained _Models/trt_model_fp32.jit.pt")
    results = {'correct': 0, 'failed': 0}
    start_time = time.time()
    with torch.no_grad():
           for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)

                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                if iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                       no_grasps=n_grasps,
                                                       grasp_width=width_img,
                                                       threshold=iou_threshold
                                                       )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1
                fig = plt.figure()
                plot_results(fig,
                        rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=n_grasps,
                        grasp_width_img=width_img
                    )
                cv2.waitKey(0)
                if vis:
                    save_results(
                        rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                        depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                        grasp_q_img=q_img,
                        grasp_angle_img=ang_img,
                        no_grasps=n_grasps,
                        grasp_width_img=width_img
                    )
        
    avg_time = (time.time() - start_time) / len(test_data)
    if iou_eval:
            print('IOU Results: %d/%d = %f' % (results['correct'],results['correct'] + results['failed'],results['correct'] / (results['correct'] + results['failed'])))
            print('average evaluation time per image: {}ms'.format(avg_time * 1000))
    del net
    torch.cuda.empty_cache()

test()