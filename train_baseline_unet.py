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

from data import transforms
from utils.math import complex_abs
from models.generator.generator_experimental import GeneratorModel
from utils.general.helper import prep_input_2_chan, readd_measures_im
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from skimage.metrics import structural_similarity

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

        z = torch.zeros((input.shape[0], 512))

        output = model(input, z)  # .squeeze(1)

        output = readd_measures_im(output, input, args)

        target_im = prep_input_2_chan(target, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)
        output_im = prep_input_2_chan(output, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)

        loss = 10 * mse(output_im, target_im) - mssim_tensor(target_im, output_im)
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

            z = torch.zeros((input.shape[0], 512))

            output = model(input, z)

            output = readd_measures_im(output, input, args)

            target_im = prep_input_2_chan(target, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)
            output_im = prep_input_2_chan(output, args.network_input, args, disc=True).to(args.device).permute(0, 2, 3, 1)

            for i in range(output_im.shape[0]):
                output = complex_abs(output_im[i])
                target = complex_abs(target_im[i])

                output = output.cpu().numpy() * std[i].numpy() + mean[i].numpy()
                target = target.cpu().numpy() * std[i].numpy() + mean[i].numpy()

                SSIM = ssim_numpy(target, output)
                losses.append(SSIM)
                psnr.append(psnr_val(target, output))

            if iter + 1 == 20:
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
    model = GeneratorModel(
        in_chans=2,
        out_chans=2,
        z_location=2,
        latent_size=512,
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
    # optimizer = torch.optim.RMSprop(params, 0.0003)
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))
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
    args.num_epochs = 50
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
