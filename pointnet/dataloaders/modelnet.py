from typing import List
import torch
import numpy as np
import h5py
import os
import os.path as osp
from utils.misc import pc_normalize


class ModelNetDataset(torch.utils.data.Dataset):
    def __init__(self, phase: str, data_dir: str):
        super().__init__()
        self.phase = phase
        self.data_dir = data_dir
        self.modelnet_dir = osp.join(data_dir, "modelnet40_ply_hdf5_2048")

        self.download_data()

        # ModelNet has only train and test splits.
        if phase == "val":
            phase = "test"

        with open(osp.join(self.modelnet_dir, f"{phase}_files.txt")) as f:
            file_list = [line.rstrip() for line in f]

        self.data = []
        self.label = []
        self.normal = []
        for fn in file_list:
            f = h5py.File(osp.join(self.modelnet_dir, osp.basename(fn)))
            self.data.append(f["data"][:])
            self.label.append(f["label"][:])
            self.normal.append(f["normal"][:])

        self.data = np.concatenate(self.data, 0).astype(np.float32)
        self.label = np.concatenate(self.label, 0).astype(np.int_)
        self.normal = np.concatenate(self.normal, 0).astype(np.float32)

    def __getitem__(self, idx):
        pc = torch.from_numpy(pc_normalize(self.data[idx]))
        label = torch.from_numpy(self.label[idx]).squeeze()

        return pc, label

    def __len__(self):
        return len(self.data)

    def download_data(self):
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not osp.exists(self.modelnet_dir):
            www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
            zipfile = osp.basename(www)
            os.system(f"wget --no-check-certificate {www}; unzip {zipfile}")
            os.system(f"mv {zipfile[:-4]} {self.data_dir}")
            os.system(f"rm {zipfile}")


def get_data_loaders(
    data_dir, batch_size, phases: List[str] = ["train", "val", "test"]
):
    datasets = []
    dataloaders = []
    for ph in phases:
        ds = ModelNetDataset(ph, data_dir)
        dl = torch.utils.data.DataLoader(
            ds, batch_size, shuffle=ph == "train", drop_last=ph == "train"
        )

        datasets.append(ds)
        dataloaders.append(dl)

    return datasets, dataloaders
