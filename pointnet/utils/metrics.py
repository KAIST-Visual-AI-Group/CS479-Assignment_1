import numpy as np
import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct = 0
        self.total = 0
        self.history = []

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        process one batch and accumulate the result.
        """
        assert preds.shape == targets.shape
        with torch.no_grad():
            c = torch.sum(preds == targets)
            t = targets.numel()

            self.correct += c
            self.total += t

        return c.float() / t

    def compute_epoch(self):
        """
        take a mean of accumulated results so far and log it into self.history
        """
        acc = self.correct.float() / self.total
        self.history.append(acc)
        self.reset()
        return acc

    def reset(self):
        self.correct = 0
        self.total = 0


class mIoU(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou_sum = 0
        self.total = 0
        self.history = []

        """
        ShapeNet Part Anno Dataset Overview

         <class name> | <class id> | <part id>
        0 Airplane    |  02691156  | [0, 1, 2, 3]
        1 Bag         |  02773838  | [4, 5]
        2 Cap         |  02954340  | [6, 7]
        3 Car         |  02958343  | [8, 9, 10, 11] 
        4 Chair       |  03001627  | [12, 13, 14 15] 
        5 Earphone    |  03261776  | [16, 17, 18]
        6 Guitar      |  03467517  | [19, 20, 21]
        7 Knife       |  03624134  | [22, 23]
        8 Lamp        |  03636649  | [24, 25, 26, 27] 
        9 Laptop      |  03642806  | [28, 29]
        10 Motorbike  |  03790512  | [30, 31, 32, 33, 34, 35] 
        11 Mug        |  03797390  | [36, 37] 
        12 Pistol     |  03948459  | [38, 39, 40] 
        13 Rocket     |  04099429  | [41, 42, 43] 
        14 Skateboard |  04225987  | [44, 45, 46] 
        15 Table      |  04379243  | [47, 48, 49]
        
        """
        self.idx2pids = {
            0: [0, 1, 2, 3],
            1: [4, 5],
            2: [6, 7],
            3: [8, 9, 10, 11],
            4: [12, 13, 14, 15],
            5: [16, 17, 18],
            6: [19, 20, 21],
            7: [22, 23],
            8: [24, 25, 26, 27],
            9: [28, 29],
            10: [30, 31, 32, 33, 34, 35],
            11: [35, 37],
            12: [38, 39, 40],
            13: [41, 42, 43],
            14: [44, 45, 46],
            15: [47, 48, 49],
        }

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, class_labels: torch.Tensor
    ):
        """
        Input:
            logits: [B, 50, num_points]
            targets: [B,num_points]
            class_labels: [B]
        Output:
            iou_per_batch
            batch_masked_pred: [B, num_points] A masked prediction where it ignores other categories' point labels when picking the highest logit.
        """
        with torch.no_grad():
            B, N = logits.shape[0], logits.shape[-1]
            device = logits.device
            batch_iou = torch.zeros(B, dtype=torch.float).to(device)
            batch_masked_pred = torch.zeros(B, N, dtype=torch.long).to(device)
            for i in range(B):
                cl = int(class_labels[i])
                pids = self.idx2pids[cl]

                logit = logits[i]
                target = targets[i]
                mask = torch.zeros_like(logit)
                mask[pids, :] = 1
                logit.masked_fill(mask == 0, -1e-9)
                masked_pred = torch.argmax(logit, dim=0)
                batch_masked_pred[i] = masked_pred

                for pid in pids:
                    pd = masked_pred == pid
                    gt = target == pid

                    union = (gt | pd).sum()
                    inter = (gt & pd).sum()

                    if union == 0:
                        batch_iou[i] += 1
                    else:
                        batch_iou[i] += inter / union

                batch_iou[i] /= len(pids)

            self.iou_sum += batch_iou.sum()
            self.total += class_labels.numel()
            
        iou_per_batch = batch_iou.sum() / class_labels.numel()
        return iou_per_batch, batch_masked_pred

    def compute_epoch(self):
        iou = self.iou_sum / self.total
        self.history.append(iou)
        self.reset()
        return iou

    def reset(self):
        self.iou_sum = 0
        self.total = 0
