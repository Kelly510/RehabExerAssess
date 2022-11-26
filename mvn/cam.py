import os
import cv2
import yaml
import numpy as np
import seaborn as sns

sns.set_theme(style='darkgrid')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn as nn
from torch.nn import functional as F
from utils.vis import draw_3d_pose, fig_to_array


def visualize_CAM(model, dataloader, device, vis_dir, vis, movement_dir):
    thres = None
    if vis:
        with open(os.path.join(movement_dir, 'thres.yaml'),
                  encoding='utf-8') as f:
            thres = yaml.load(f.read(), Loader=yaml.FullLoader)
        os.makedirs(vis_dir, exist_ok=True)

    model.eval()
    fmap_block = []
    grad_block = []

    def forward_hook(module, input, output):
        fmap_block.append(output)

    def backward_hook(module, input, output):
        grad_block.append(output[0].detach())

    model.st_gcn_networks[2].relu.register_forward_hook(forward_hook)
    model.st_gcn_networks[2].relu.register_backward_hook(backward_hook)

    pos_cam_value_list = []
    neg_cam_value_list = []
    for i, (samples_batch, labels_batch) in enumerate(dataloader):
        batch_size = labels_batch.shape[0]
        samples_batch = samples_batch.float().to(device)
        labels_batch_binary = (labels_batch) != 0

        for j in range(batch_size):
            score = model(samples_batch[j:j + 1])
            label = torch.argmax(score[0]).cpu()
            model.zero_grad()
            score[0, 1].backward()

            fmaps_val = fmap_block[0].cpu().squeeze()
            grads_val = grad_block[0].cpu().squeeze()

            weights = torch.mean(grads_val, dim=(1, 2))
            cam = torch.einsum('c,cnj->nj', (weights, fmaps_val))
            cam = F.relu(cam)
            cam = cam.float().detach().numpy()

            if label == 1:
                pos_cam_value_list.append(np.max(cam))
            else:
                neg_cam_value_list.append(np.max(cam))

            if vis:
                skeleton = samples_batch[j].cpu().numpy()
                filename = os.path.join(
                    vis_dir,
                    '{}_{}.avi'.format(i * batch_size + j,
                                       'positive' if label else 'negative'))
                plot_cam(skeleton, cam, filename,
                         'positive' if label else 'negative',
                         label == labels_batch_binary[j],
                         dataloader.dataset.dataset_fps // 2,
                         dataloader.dataset.connectivity, thres)

            fmap_block.clear()
            grad_block.clear()

    OvR = compute_distribution_OvR(pos_cam_value_list, neg_cam_value_list,
                                   os.path.join(vis_dir, 'dist.jpg'))
    return OvR


def plot_cam(input,
             cam,
             filename,
             label,
             correct,
             fps=None,
             connectivity=None,
             thres=[0.0075, 0.01]):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (640, 480), True)

    for t in range(input.shape[0]):
        fig = plt.figure()
        ax = Axes3D(fig)
        blue_cam = np.clip(cam[t], a_min=0, a_max=thres[0])
        blue_cam = (thres[0] - blue_cam) / (thres[0] + 1e-5)
        blue_mask = (cam[t] < thres[0])
        red_cam = np.clip(cam[t], a_min=thres[0], a_max=thres[1])
        red_cam = (red_cam - thres[0]) / (thres[1] - thres[0] + 1e-5)
        red_mask = (cam[t] >= thres[0])

        color = np.stack([
            red_cam * red_mask,
            np.zeros_like(red_mask), blue_cam * blue_mask
        ],
                         axis=1)
        keypoints = input[t]
        try:
            draw_3d_pose(keypoints,
                         ax,
                         radius=1,
                         c=color,
                         point_size=15,
                         line_width=2,
                         connectivity=connectivity)
            kpts_fig = np.ascontiguousarray(fig_to_array(fig)[..., (2, 1, 0)])
            cv2.putText(kpts_fig, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0) if correct else (0, 0, 255), 1)
            writer.write(kpts_fig)
        except ValueError:
            print(color)

        plt.clf()
        plt.close('all')

    writer.release()


def compute_distribution_OvR(pos_cam_list, neg_cam_list, filename):
    plt.figure()

    try:
        pos_cam_value = np.stack(pos_cam_list, axis=0).reshape(-1)
        sns.kdeplot(pos_cam_value,
                    clip=[0, 0.8],
                    shade=True,
                    color='r',
                    label='positive',
                    alpha=.5)
    except Exception as e:
        print(e)
        return 0

    try:
        neg_cam_value = np.stack(neg_cam_list, axis=0).reshape(-1)
        sns.kdeplot(neg_cam_value,
                    clip=[0, 0.8],
                    shade=True,
                    color='b',
                    label='negative',
                    alpha=.5)
    except Exception as e:
        print(e)
        return 0

    x_array = np.linspace(0, 1.5 * np.max(pos_cam_value), 100)
    pos_cam_kde = get_kde(x_array, pos_cam_value)
    neg_cam_kde = get_kde(x_array, neg_cam_value)

    cam_kde_stack = np.stack([pos_cam_kde, neg_cam_kde])
    inter = np.trapz(np.min(cam_kde_stack, axis=0), x_array)
    union = np.trapz(np.max(cam_kde_stack, axis=0), x_array)
    OvR = inter / union

    plt.title('CAM Distribution - OvR: {:.4f}'.format(OvR))
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(filename)

    plt.clf()
    plt.close('all')

    return OvR


def get_kde(x_array, data_array):
    bandwidth = 1.05 * np.std(data_array) * (data_array.shape[0]**(-1 / 5))

    def gauss(x):
        import math
        return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x**2))

    N = data_array.shape[0]
    y_array = np.zeros_like(x_array)
    if N == 0:
        return y_array
    for i in range(x_array.shape[0]):
        for j in range(N):
            y_array[i] += gauss((x_array[i] - data_array[j]) / bandwidth)
        y_array[i] /= (N * bandwidth)

    return y_array
