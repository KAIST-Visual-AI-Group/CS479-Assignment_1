import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import json

def pc_normalize(pc: np.ndarray):
    m = pc.mean(0)
    pc = pc - m
    s = np.max(np.sqrt(np.sum(pc**2, -1)))
    pc = pc / s

    return pc



def save_samples(pointclouds: torch.Tensor, groundtruths: torch.Tensor, preds: torch.Tensor, filename: str):
    """
    pointclouds: [num_sample, num_points, 3]
    groundtruths: [num_sample, num_points]
    preds: [num_sample, num_points]
    filename: output filename
    """
    fin = open("data/shapenet_part_seg_hdf5_data/part_color_mapping.json")
    color_maps = np.array(json.load(fin))
    color_mapping = lambda x : color_maps[x]
    fin.close()

    assert pointclouds.shape[:2] == groundtruths.shape == preds.shape

    num_sample = pointclouds.shape[0]
    pcs = pointclouds.clone().detach().cpu().numpy()
    gts = groundtruths.clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu().numpy()

    fig = plt.figure(figsize=(2*4, num_sample*4))

    for i in range(num_sample):
        ax = fig.add_subplot(num_sample, 2, 2*i+1, projection="3d")
        ax.scatter(pcs[i,:,0], pcs[i,:,2], pcs[i,:,1], c=color_mapping(gts[i]))

        ax.set_xlim(-.7, .7)
        ax.set_ylim(-.7, .7)
        ax.set_zlim(-.7, .7)
        ax.axis("off")

        ax = fig.add_subplot(num_sample, 2, 2*i+2, projection="3d")
        ax.scatter(pcs[i,:,0], pcs[i,:,2], pcs[i,:,1], c=color_mapping(preds[i]))

        ax.set_xlim(-.7, .7)
        ax.set_ylim(-.7, .7)
        ax.set_zlim(-.7, .7)
        ax.axis("off")

    plt.tight_layout()
    fig.suptitle("Left: Groundtruths  Right: Predictions", fontsize=18)
    plt.savefig(filename)
        
