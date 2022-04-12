import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import optim

from losses.loss import MSELoss
from models.model import UNet
from data.transforms import get_transforms
from data.mura_dataset import MURADataset
from config.train_args import train_options
from utils.metrics import AvgMeter, get_metrics


def get_loaders(options, train_df, valid_df):

    train_dataset = MURADataset(
        train_df,
        f"{options.data_path}/{'AP/train'}",
        heatmap=options.heatmap,
        coefficient=options.coefficient,
        transforms=get_transforms(
            intensity=options.intensity, mode="train", size=options.size
        ),
        size=options.size,
        point_count=options.point_count,
    )

    valid_dataset = MURADataset(
        valid_df,
        f"{options.data_path}/{'AP/train'}",
        heatmap=options.heatmap,
        coefficient=options.coefficient,
        transforms=get_transforms(
            intensity=options.intensity, mode="valid", size=options.size
        ),
        size=options.size,
        point_count=options.point_count,
    )

    batch_size = 8 if options.size == 512 else options.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=options.num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=options.num_workers,
    )

    return train_loader, valid_loader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def one_epoch(
    model, criterion, loader, device, optimizer=None, lr_scheduler=None, mode="train",
):
    loss_meter = AvgMeter()
    mae_meter = AvgMeter()
    alpha_meter = AvgMeter()

    for xb, heatmaps, points in tqdm(loader):
        xb, heatmaps, points = xb.to(device), heatmaps.to(device), points.to(device)
        preds = model(xb)
        loss = criterion(preds, heatmaps)
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if isinstance(lr_scheduler, optim.lr_scheduler.OneCycleLR):
                lr_scheduler.step()

        count = xb.size(0)
        loss_meter.update(loss.item(), count)
        mae, alpha_error = get_metrics(preds.detach().cpu(), points.cpu())
        mae_meter.update(mae, count)
        alpha_meter.update(alpha_error, count)

    return loss_meter, mae_meter, alpha_meter

def predict(model, loader, device):
    model.eval()
    xb, heatmaps, points = next(iter(loader))
    with torch.no_grad():
        heatmap_preds = model(xb.to(device)).detach().cpu()

    return xb, heatmap_preds, heatmaps, points

def train_eval(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    options,
    lr_scheduler=None,
):

    best_loss = float("inf")
    best_mae = float("inf")
    reduce_coeff = False
    reduce_lr = False
    best_alpha = float("inf")

    for epoch in range(options.epochs):
        print("*" * 30)
        print(f"Epoch {epoch + 1}")
        current_lr = get_lr(optimizer)
        print(f"Current Learning Rate: {current_lr:.4f}")

        model.train()
        train_loss, train_mae, train_alpha_error = one_epoch(
            model,
            criterion,
            train_loader,
            device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            mode="train",
        )
        model.eval()
        with torch.no_grad():
            valid_loss, valid_mae, valid_alpha_error = one_epoch(
                model,
                criterion,
                valid_loader,
                device,
                optimizer=None,
                lr_scheduler=None,
                mode="valid",
            )

        # inputs, heatmap_preds, targets, points = predict(model, valid_loader, device)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f"{options.model_path}/best_loss_fold_{options.fold}.pt")
            print("Saved best loss model!")

        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(valid_loss.avg)
            if current_lr != get_lr(optimizer):
                print("Loading best model weights!")
                model.load_state_dict(
                    torch.load(f"{options.model_path}/best_loss_fold_{options.fold}.pt", map_location=device)
                )

        if isinstance(lr_scheduler, optim.lr_scheduler.CosineAnnealingLR):
            lr_scheduler.step()

        if valid_alpha_error.avg < best_alpha:
            best_alpha = valid_alpha_error.avg
            torch.save(
                model.state_dict(),
                f"{options.model_path}/best_model_alpha_fold_{options.fold}.pt",
            )

        print(f"Loss: {valid_loss.avg:.5f}")
        print(f"MAE: {valid_mae.avg:.5f}")
        print("*" * 30)


def main(parser):
    options = parser.parse_args()

    for i, arg in enumerate(vars(options)):
        print(f"{i + 1}- {arg}, {getattr(options, arg)}")

    device = torch.device("cuda")

    df = pd.read_csv(f"{options.data_path}/{options.df}")
    valid_df = df[df["fold"] == options.fold]
    train_df = df[df["fold"] != options.fold]

    train_loader, valid_loader = get_loaders(options, train_df, valid_df)

    model = UNet(
        model=options.model, pretrained="imagenet", classes=options.point_count
    )

    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )

    if options.lr_scheduler == "ReduceLR":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    elif options.lr_scheduler == "OneCycle":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            pct_start=0.1,
            div_factor=25.0,
            max_lr=options.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=options.epochs,
        )
    elif options.lr_scheduler == "Cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=5, eta_min=1e-6, last_epoch=-1,
        )
    else:
        lr_scheduler = None

    criterion = MSELoss(weighted=True)

    train_eval(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        lr_scheduler=lr_scheduler,
        options=options,
    )

parser = train_options()
if __name__ == "__main__":
    main(parser=parser)
    
