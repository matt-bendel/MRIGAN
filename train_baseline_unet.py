"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# Author : Saurav
# Code to train k-space U-Net with a skip connection to make the network learn only unmasked-kspace
# Specifications:
# Brain Data
# GRO Sampling Pattern
# acceleration R = 4
# Approximate noise lvl = 20 dB (need to calculate average accurately)
# Validation Data from fastMRI Brain data with more than 10 coils were reduced to 8 coils
# The Validation Data was split into 70% training, 10% for testing and validation, 20% for computing precision values
# l1 loss was first used
# Complex data was fed as separate channels making the U-Net to have 8*2 =16 input channels of undersampled k-space data
# Output channels is also fully sampled 16 channels k-space data
# Input and output was subtracted using a mean value of (mean of means: tensor(-4.0156e-11)) and divided by a standard deviation of (mean of stds: tensor(2.5036e-05))
# Therefore to use this U-Net output in PnP, multiply the std and add the mean menitoned above.


# PYTHONPATH=. python models/unet/kspace_UNET_brain_mri_with_skip_connection.py --data-path '/storage/fastMRI_brain/data/multicoil_val' --data-parallel


import logging
import pathlib
import random
import shutil
import time
import os
import pytorch_ssim
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import matplotlib.pyplot as plt

from typing import Optional
from data import transforms
from utils.math import complex_abs
from models.baseline_unet.unet_residual import UnetModelRes
from utils.general.helper import prep_input_2_chan, readd_measures_im
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mssim_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    ssim_loss = pytorch_ssim.SSIM()
    return ssim_loss(gt, pred)

def psnr_val(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val

def ssim_numpy(gt, pred):
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max()

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim


def l1_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(gt - pred) ** 2 / torch.norm(gt) ** 2


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, mean, std, nnz_index_mask = data
        input = input.to(args.device)
        target = target.to(args.device)

        input = prep_input_2_chan(input, args.network_input, args)
        target = prep_input_2_chan(target, args.network_input, args)

        output = model(input)  # .squeeze(1)

        # output = readd_measures_im(output, input, args)

        target_im = prep_input_2_chan(target, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)
        output_im = prep_input_2_chan(output, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)

        loss = 0.001 * F.l1_loss(target_im, output_im) - mssim_tensor(target_im, output_im)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()

        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )

        start_iter = time.perf_counter()

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    psnr = []
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, nnz_index_mask = data
            input = input.to(args.device)
            target = target.to(args.device)

            input = prep_input_2_chan(input, args.network_input, args)
            target = prep_input_2_chan(target, args.network_input, args)

            output = model(input)

            # output = readd_measures_im(output, input, args)

            target_im = prep_input_2_chan(target, args.network_input, args, disc=True).to(args.device)
            output_im = prep_input_2_chan(output, args.network_input, args, disc=True).to(args.device)

            for i in range(output_im.shape[0]):
                output_rss = torch.zeros(8, output_im.shape[2], output_im.shape[2], 2)
                output_rss[:, :, :, 0] = output_im[i, 0:8, :, :]
                output_rss[:, :, :, 1] = output_im[i, 8:16, :, :]
                output = transforms.root_sum_of_squares(complex_abs(output_rss * std[i] + mean[i]))

                target_rss = torch.zeros(8, target_im.shape[2], target_im.shape[2], 2)
                target_rss[:, :, :, 0] = target_im[i, 0:8, :, :]
                target_rss[:, :, :, 1] = target_im[i, 8:16, :, :]
                target = transforms.root_sum_of_squares(complex_abs(target_rss * std[i] + mean[i]))

                output = output.cpu().numpy()
                target = target.cpu().numpy()

                SSIM = ssim_numpy(target, output)
                losses.append(SSIM)
                psnr.append(psnr_val(target, output))
                if iter+1==1 and i==2:
                    plt.figure()
                    plt.imshow(np.abs(output), cmap='gray')
                    plt.savefig('temp_out.png')

                    plt.figure()
                    plt.imshow(np.abs(target), cmap='gray')
                    plt.savefig('temp_targ.png')

            if iter + 1 == 40:
                break

        writer.add_scalar('DevSSIM:', np.mean(losses), epoch)
        print(f'PSNR: {np.mean(psnr):.2f}')

    return np.mean(losses), time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = UnetModelRes(
        in_chans=16,
        out_chans=16,
        chans=256,
        num_pool_layers=4
    ).to(torch.device('cuda'))
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, 0.0003)
    # optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))
    return optimizer


def main(args):
    args.exp_dir = pathlib.Path(f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/baseline/{args.network_input}')
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        args.checkpoint = pathlib.Path(f'/home/bendel.8/trained_models/baseline/{args.network_input}/model.pt')
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        print('new')
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader = create_data_loaders(args)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)

        is_new_best = -dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, -dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'SSIM = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


if __name__ == '__main__':
    mse = torch.nn.MSELoss()
    args = create_arg_parser().parse_args()
    args.num_epochs = 100
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
