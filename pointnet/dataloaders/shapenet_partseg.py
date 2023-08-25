from typing import List
import torch
import numpy as np
import h5py
import os
import os.path as osp
from utils.misc import pc_normalize


class ShapeNetPartSegDataset(torch.utils.data.Dataset):
    def __init__(self, phase: str, data_dir: str):
        super().__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.shapenet_dir = osp.join(data_dir, "shapenet_part_seg_hdf5_data")

        self.download_data()

        with open(osp.join(self.shapenet_dir, f"{phase}_hdf5_file_list.txt")) as f:
            file_list = [line.rstrip() for line in f]
        
        self.data = []
        self.pc_label = []
        self.class_label = []
        for fn in file_list:
            f = h5py.File(osp.join(self.shapenet_dir, fn))
            self.data.append(f["data"][:])
            self.pc_label.append(f["pid"][:])
            self.class_label.append(f["label"][:])

        self.data = np.concatenate(self.data, 0).astype(np.float32)
        self.pc_label = np.concatenate(self.pc_label, 0).astype(np.int_)
        self.class_label = np.concatenate(self.class_label, 0).astype(np.int_)

    def __getitem__(self, idx):
        pc = torch.from_numpy(pc_normalize(self.data[idx]))
        pc_label = torch.from_numpy(self.pc_label[idx])
        class_label = torch.from_numpy(self.class_label[idx]).squeeze()
        return pc, pc_label, class_label

    def __len__(self):
        return len(self.data)

    def download_data(self):
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not osp.exists(self.shapenet_dir):
            www = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
            zipfile = osp.basename(www)
            os.system(f"wget --no-check-certificate {www}; unzip {zipfile}")
            os.system(f"mv hdf5_data {self.shapenet_dir}")
            os.system(f"rm {zipfile}")

def get_data_loaders(
    data_dir, batch_size, phases: List[str] = ["train", "val", "test"]
):
    datasets = []
    dataloaders = []
    for ph in phases:
        ds = ShapeNetPartSegDataset(ph, data_dir)
        dl = torch.utils.data.DataLoader(
            ds, batch_size, shuffle=ph == "train", drop_last=ph == "train"
        )

        datasets.append(ds)
        dataloaders.append(dl)

    return datasets, dataloaders
