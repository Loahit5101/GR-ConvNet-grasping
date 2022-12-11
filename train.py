import json
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from network import GRconvNet
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import statistics

def post_process_output(q_img, cos_img, sin_img, width_img):

    q_img = q_img.cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
    width_img = width_img.cpu().numpy().squeeze() * 150.0

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def run():
    if torch.cuda.is_available():
        
        device = torch.device("cuda")
    Dataset = get_dataset("cornell")
    dataset = Dataset('/home/loahit/Downloads/archive',output_size=224,ds_rotate=True,random_rotate=True,random_zoom=True,include_depth=True,include_rgb=True)

    print(dataset.length)
    indices = list(range(dataset.length))
    split = int(np.floor(0.9 * dataset.length))
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    batch_size=8
    num_workers=8

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )

    network = GRconvNet()

    network = network.to(device)
    optimizer = optim.Adam(network.parameters())
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    summary(network, (4, 224, 224))
    epochs=50
    best_iou = 0.0

    iou_threshold=0.25
    batch_idx=0
    train_losses = []
    valid_losses = []
    val_ious=[]
    train_iou=[]
    for epoch in range(epochs):

        network.train()

        train_results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
        }
        train_batch_losses = []
        for x, y, _, _, _ in train_data:
                batch_idx=batch_idx+1
                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                loss_dict = network.compute_loss(xc, yc)
                
                loss = loss_dict['loss']
                
                train_batch_losses.append(loss.item())
                
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                vis=False
                if vis:
                    imgs = []
                    n_img = min(4, x.shape[0])
                    for idx in range(n_img):
                        imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                            x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      loss_dict['pred'].values()])
                        gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                        cv2.waitKey(2)
        train_losses.append(statistics.mean(train_batch_losses))
        results_val = {
                             'correct': 0,
                            'failed': 0,
                            'loss': 0,
                            'losses': {

                            }
                          }
        ld = len(val_data)

        network.eval()
        val_results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
        }

        val_batch_losses = []
        with torch.no_grad():
                   for x, y, didx, rot, zoom_factor in val_data:
                        xc = x.to(device)
                        yc = [yy.to(device) for yy in y]
                        loss_dict_val = network.compute_loss(xc, yc)
                        
                        loss_val = loss_dict_val['loss']
                        val_batch_losses.append(loss_val.item())
                        q_out, ang_out, w_out = post_process_output(loss_dict_val['pred']['pos'], loss_dict_val['pred']['cos'],
                                                        loss_dict_val['pred']['sin'], loss_dict_val['pred']['width'])

                        s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold)
                        if s:
                            val_results['correct'] += 1
                        else:
                            val_results['failed'] += 1
        valid_losses.append(statistics.mean(val_batch_losses))
        val_iou = val_results['correct'] / (val_results['correct'] + val_results['failed'])
        val_ious.append(val_iou)
        if val_iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(network, 'epoch_%02d_iou_%0.2f' % (epoch, val_iou))
            best_iou = val_iou
    
    epochs_list = range(1, len(val_ious) + 1)
    plt.plot(epochs_list, val_ious, 'b', label='Validation IoUs')
    plt.title('Validation IoU')
    plt.legend()
    plt.savefig("Val_iou.png")
    plt.show()
    plt.figure()

    plt.plot(epochs_list, train_losses, 'b', label='Training Loss')
    plt.plot(epochs_list, valid_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Losses.png")
    plt.show()


run()
