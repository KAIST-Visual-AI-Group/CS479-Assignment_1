import torch
import torch.nn.functional as F
import argparse
from datetime import datetime
from tqdm import tqdm
from model import PointNetAutoEncoder
from dataloaders.modelnet import get_data_loaders
from utils.metrics import Accuracy
from utils.model_checkpoint import CheckpointManager
from pytorch3d.loss.chamfer import chamfer_distance

 

def step(points, model):
    """
    Input : 
        - points [B, N, 3]
    Output : loss
        - loss []
        - preds [B, N, 3]
    """

    # TODO : Implement step function for AutoEncoder. 
    # Hint : Use chamferDist defined in above
    # Hint : You can compute chamfer distance between two point cloud pc1 and pc2 by chamfer_distance(pc1, pc2)
    
    preds = None
    loss = None

    return loss, preds


def train_step(points, model, optimizer):
    loss, preds = step(points, model)

    # TODO : Implement backpropagation using optimizer and loss

    return loss, preds


def validation_step(points, model):
    loss, preds = step(points, model)

    return loss, preds


def main(args):
    global device
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    model = PointNetAutoEncoder(num_points=2048)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 80], gamma=0.5
    )

    # automatically save only topk checkpoints.
    if args.save:
        checkpoint_manager = CheckpointManager(
            dirpath=datetime.now().strftime("checkpoints/auto_encoding/%m-%d_%H-%M-%S"),
            metric_name="val_loss",
            mode="min",
            topk=2,
            verbose=True,
        )

    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./data", batch_size=args.batch_size, phases=["train", "val", "test"]
    )

    for epoch in range(args.epochs):

        # training step
        model.train()
        pbar = tqdm(train_dl)
        train_epoch_loss = []
        for points, _ in pbar:
            train_batch_loss, train_batch_preds = train_step(points, model, optimizer)
            train_epoch_loss.append(train_batch_loss)
            pbar.set_description(
                f"{epoch+1}/{args.epochs} epoch | loss: {train_batch_loss:.4f}"
            )

        train_epoch_loss = sum(train_epoch_loss) / len(train_epoch_loss)

        # validataion step
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            for points, _ in val_dl:
                val_batch_loss, val_batch_preds = validation_step(points, model)
                val_epoch_loss.append(val_batch_loss)

            val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            print(
                f"train loss: {train_epoch_loss:.4f} | val loss: {val_epoch_loss:.4f}"
            )

        if args.save:
            checkpoint_manager.update(model, epoch, round(val_epoch_loss.item(), 4), f"AutoEncoding_ckpt")

        scheduler.step()

    if args.save:
        checkpoint_manager.load_best_ckpt(model, device)
    model.eval()
    with torch.no_grad():
        test_epoch_loss = []
        for points, _ in test_dl:
            test_batch_loss, test_batch_preds = validation_step(points, model)
            test_epoch_loss.append(test_batch_loss)

        test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
        print(f"test loss: {test_epoch_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet ModelNet40 AutoEncoder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    args.gpu = 0
    args.save = True

    main(args)
