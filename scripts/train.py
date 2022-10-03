"""
Example script for training SPHARM-Net.

If you use this code, please cite the following paper.

    Seungbo Ha and Ilwoo Lyu
    SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation.
    IEEE Transactions on Medical Imaging, 41(10), 2739-2751, 2022

Copyright 2022 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from spharmnet import SPHARM_Net
from spharmnet.lib.utils import SphericalDataset, Logger, eval_accuracy, eval_dice
from spharmnet.lib.loss import DiceLoss
from spharmnet.lib.io import read_mesh


def get_args():
    parser = argparse.ArgumentParser()

    # Dataset & dataloader
    parser.add_argument("--sphere", type=str, default="./sphere/ico6.vtk", help="Sphere mesh (vtk or FreeSurfer format)")
    parser.add_argument("--data-dir", type=str, default="./dataset", help="Path to re-tessellated data")
    parser.add_argument("--data-norm", action="store_true", help="Z-score+prctile data normalization")
    parser.add_argument("--preload", type=str, choices=["none", "cpu", "device"], default="device", help="Data preloading")
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+", help="List of geometry")
    parser.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for learning", required=True)
    parser.add_argument("--n-splits", type=int, default=5, required=False, help="A total of cross-validation folds")
    parser.add_argument("--fold", type=int, default=1, required=False, help="Cross-validation fold")
    parser.add_argument("--classes", type=int, nargs="+", help="List of regions of interest")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
    parser.add_argument("--aug", type=int, default=0, help="Level of data augmentation")

    # Training and evaluation
    parser.add_argument("--epochs", type=int, default=20, help="Max epoch")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--no-decay", action="store_true", help="Disable decay (every 2 epochs if no progress)")
    parser.add_argument("--loss", type=str, default="dl", choices=["dl", "ce"], help="dl: Dice loss, ce: cross entropy")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to the log files (output)")
    parser.add_argument("--ckpt-dir", type=str, default="./logs", help="Path to the checkpoint file (output)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint (pth) to resume training")

    # SPHARM-Net settings
    parser.add_argument("-D", "--depth", type=int, default=3, help="Depth of SPHARM-Net")
    parser.add_argument("-C", "--channel", type=int, default=128, help="# of channels in the entry layer of SPHARM-Net")
    parser.add_argument("-L", "--bandwidth", type=int, default=80, help="Bandwidth of SPHARM-Net")
    parser.add_argument("--interval", type=int, default=5, help="Anchor interval of hamonic coefficients")

    # Machine settings
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for training (normally, starting with 0)")
    parser.add_argument("--no-cuda", action="store_true", help="No CUDA")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for basis reconstruction")

    args = parser.parse_args()

    return args


def step(model, train_loader, device, criterion, epoch, logger, nclass, optimizer=None, pbar=False):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    progress = tqdm if pbar else lambda x: x

    running_loss = 0.0
    total_correct = 0  # for calculating accuracy
    total_vertex = 0  # for calculating accuracy

    total_dice = torch.empty((0, nclass))

    iter = 0
    with torch.no_grad() if optimizer is None else ExitStack():
        for input, label, _ in progress(train_loader):
            input = input.to(device)
            label = label.to(device)

            if optimizer is not None:
                optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            running_loss += loss
            _, output = torch.max(output, 1)

            correct, num_vertex = eval_accuracy(output, label)
            total_correct += correct
            total_vertex += num_vertex

            batch_dice = eval_dice(output, label, nclass)  # batch_dice : [batch, nclass]
            total_dice = torch.cat([total_dice, batch_dice], dim=0)

            iter += 1

            if optimizer is not None:
                loss.backward()
                optimizer.step()

    accuracy = total_correct / total_vertex

    logger.write([epoch + 1, running_loss.item() / iter, accuracy, torch.mean(total_dice).item()])

    return accuracy


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    preload = None if args.preload == "none" else device if args.preload == "device" else args.preload

    if not args.cuda:
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("Loading data...")
    sphere = os.path.join(args.sphere)
    v, _ = read_mesh(sphere)

    # dataset partition
    partition = ["train", "val", "test"]
    dataset = dict()
    for partition_type in partition:
        dataset[partition_type] = SphericalDataset(
            data_dir=args.data_dir,
            partition=partition_type,
            fold=args.fold,
            num_vert=v.shape[0],
            in_ch=args.in_ch,
            classes=args.classes,
            seed=args.seed,
            aug=args.aug,
            n_splits=args.n_splits,
            hemi=args.hemi,
            data_norm=args.data_norm,
            preload=preload,
        )

    # dataset loader
    loader = dict()
    loader["train"] = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True, drop_last=False)
    loader["val"] = DataLoader(dataset["val"], batch_size=args.batch_size, shuffle=False, drop_last=False)
    loader["test"] = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False, drop_last=False)

    # logger
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = dict()
    for partition_type in partition:
        logger[partition_type] = Logger(os.path.join(args.log_dir, partition_type + ".log"))

    # SPHARM-Net
    print("Loading model...")
    start_epoch = 0
    model = SPHARM_Net(
        sphere=sphere,
        device=device,
        in_ch=len(args.in_ch),
        n_class=len(args.classes),
        C=args.channel,
        L=args.bandwidth,
        D=args.depth,
        interval=args.interval,
        threads=args.threads,
        verbose=True,
    )
    model.to(device)

    # model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num of params", params)
    arguments = args.__dict__
    arguments["num_params"] = params

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=1, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-8
    )

    # training loss
    assert args.loss in ["ce", "dl"]
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "dl":
        criterion = DiceLoss()

    # resume if past training is available
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # ckpt path
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # logging the current configurations
    for partition_type in partition:
        if partition_type == "train":
            logger[partition_type].write(arguments)
        logger[partition_type].write({"fold_data": dataset[partition_type].subj_list()})
        logger[partition_type].write(
            ["{:10}".format("Epoch"), "{:15}".format("Loss"), "{:15}".format("Accuracy"), "{:15}".format("Dice")]
        )

    # main loop
    best_acc = 0
    for epoch in range(start_epoch, args.epochs):
        step(model, loader["train"], device, criterion, epoch, logger["train"], len(args.classes), optimizer, True)
        val_acc = step(model, loader["val"], device, criterion, epoch, logger["val"], len(args.classes))

        if not args.no_decay:
            scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "acc": best_acc,
                    "args": arguments,
                },
                os.path.join(args.ckpt_dir, "best_model_fold{}.pth".format(args.fold)),
            )

    # test
    test_ckpt = torch.load(os.path.join(args.ckpt_dir, "best_model_fold{}.pth".format(args.fold)))
    model.load_state_dict(test_ckpt["model_state_dict"])
    model.to(device)
    step(model, loader["test"], device, criterion, test_ckpt["epoch"], logger["test"], len(args.classes))


if __name__ == "__main__":
    args = get_args()
    main(args)
