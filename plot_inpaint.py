# TODO: PLOT PSNR VS NUM CODE VECTORS
# TODO: PLOT SSIM VS NUM CODE VECTORS
# TODO: PLOT 5 DIFFERENT RECONSTRUCTIONS FROM EACH METHOD W/ CORRESPONDING UNCERTAINTY MAP ALL ON SAME SCALE
import time
import pickle
import random
import os
import torch
import pathlib
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

from data import transforms
from typing import Optional
from utils.math import complex_abs
from utils.training.prepare_data_inpaint import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start, build_model
from utils.general.helper import readd_measures_im, prep_input_2_chan
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim


class GANS:
    def __init__(self, args):
        self.args = args
        self.gen = self.get_gen(self.args)
        self.gen.eval()

    def get_gen(self, args):
        checkpoint_file_gen = pathlib.Path(
            f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/inpaint/generator_best_model.pt')
        checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

        generator = build_model(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)

        generator.load_state_dict(checkpoint_gen['model'])
        generator.eval()

        return generator

    def get_noise(self, batch_size):
        # change the noise dimension as required
        z = torch.FloatTensor(np.random.normal(size=(batch_size, self.args.latent_size), scale=np.sqrt(1))).to(
            self.args.device)
        return z

    def compute_std_dev(self, recons, mean):
        std_dev = np.zeros(mean.shape)
        for val in recons:
            std_dev = std_dev + np.power((val - mean), 2)

        std_dev = std_dev / args.num_z
        return np.sqrt(std_dev)

    def __call__(self, y, target, mean, std, plot_num):
        recons = []
        inds = torch.nonzero(y == 0)

        batch_size = y.size(0)
        avg_tensor = torch.zeros(8, 128, 128).to(self.args.device)
        for j in range(8):
            z = self.get_noise(batch_size)
            samples = self.gen(y, z)
            samples[inds] = target[inds]
            avg_tensor[j, :, :] = samples[plot_num, :, :, :].squeeze(0) * std[plot_num] + mean[plot_num]

            recons.append(avg_tensor[j, :, :].cpu().numpy())

        avg = torch.mean(avg_tensor, dim=0)
        std_dev = self.compute_std_dev(recons, avg.cpu().numpy())

        return recons, avg, std_dev


def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if method != None:
            ax.set_title(method, size=10)

        ax.text(1, 0.85, f'PSNR: {psnr_val:.2f}\nSNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='center', fontsize='xx-small', color='yellow')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis', vmin=0, vmax=3e-5)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        ax.set_title(method, size=10)
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])

    return im, ax


def generate_error_map(fig, target, recon, image_ind, rows, cols, relative=False, k=1, kspace=False, title=None):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    if kspace:
        recon = recon ** 0.4
        target = target ** 0.4

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet', vmax=0.0001)

    if title != None:
        ax.set_title(title, size=10)
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def get_colorbar(fig, im, ax, left=False):
    fig.subplots_adjust(right=0.85)  # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.01
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10]) if not left else fig.add_axes(
        [x10 - 2 * pad, y10, width, y11 - y10])

    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2e')  # Generate colorbar
    cbar.ax.tick_params(labelsize=8)

    if left:
        cbar_ax.yaxis.tick_left()
        cbar_ax.yaxis.set_label_position('left')


# TODO: UPDATE ALL PLOTTING SCRIPTS W/ APPROPRIATE STUFF
def create_mean_error_plots(generator, input, target, mean, std):
    num_rows = 3
    num_cols = 8

    inds = [2, 3, 4, 5]
    avg = []
    std_devs = []
    gts = []
    recons = []
    for i in inds:
        recon, temp, std_dev = generator(input, target, mean, std, i)
        recons.append(recon)
        avg.append(temp.cpu().numpy())
        std_devs.append(std_dev)
        gts.append(target[i].squeeze(0).cpu().numpy() * std[i].numpy() + mean[i].numpy())

    fig = plt.figure(figsize=(5 * 2.67, 5))
    fig.subplots_adjust(wspace=0, hspace=0.05)
    im_er, ax_er = None, None
    im_std, ax_std = None, None
    for i, data in enumerate(avg):
        gt = gts[i]
        gt_ind = 1 if i == 0 else (3 if i == 1 else (5 if i == 2 else 7))
        recon_ind = 2 if i == 0 else (4 if i == 1 else (6 if i == 2 else 8))
        generate_image(fig, gt, gt, 'GT', gt_ind, num_rows, num_cols)
        generate_image(fig, gt, data, 'Recon', recon_ind, num_rows, num_cols)
        if i == 0:
            im_er, ax_er = generate_error_map(fig, gt, data, recon_ind + 8, num_rows, num_cols)
            im_std, ax_std = generate_image(fig, gt, std_devs[i], 'Std. Dev', recon_ind + 16, num_rows, num_cols)
        else:
            generate_error_map(fig, gt, data, recon_ind + 8, num_rows, num_cols)
            generate_image(fig, gt, std_devs[i], 'Std. Dev', recon_ind + 16, num_rows, num_cols)

    get_colorbar(fig, im_er, ax_er, left=True)
    get_colorbar(fig, im_std, ax_std, left=True)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/inpaint_plots/mean_error.png')

    return recons, gts


# def create_z_compare_plots(generator, input, target):
#     num_rows = 5
#     num_cols = 7
#
#     fig = plt.figure(figsize=(7 * 1.4, 7))
#     fig.subplots_adjust(wspace=0, hspace=0.05)
#     generate_image(fig, gt, gt, 'GT', 1, num_rows, num_cols)
#
#     labels = ['Adv. Only', '+Supervised', '+DC', '+Var Loss', '+DI - No DC', '+DI - w/ DC']
#
#     for i in range(num_cols - 1):
#         generate_error_map(fig, gt, recons[f'g{i + 1}'][0], i + 2, num_rows, num_cols, title=labels[i])
#         generate_error_map(fig, gt, recons[f'g{i + 1}'][1], i + 9, num_rows, num_cols)
#         generate_error_map(fig, gt, recons[f'g{i + 1}'][2], i + 16, num_rows, num_cols)
#         generate_error_map(fig, gt, recons[f'g{i + 1}'][3], i + 23, num_rows, num_cols)
#         generate_error_map(fig, gt, recons[f'g{i + 1}'][4], i + 30, num_rows, num_cols)
#
#     plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/ablation_plots/5_z.png')


def generate_gifs(recons, gts):
    reformatted_recons = {
        'z1': [],
        'z2': [],
        'z3': [],
        'z4': [],
        'z5': [],
        'z6': [],
        'z7': [],
        'z8': [],
    }
    for j in range(len(recons[0])):
        reformatted_recons[f'z{j+1}'].append(recons[0][j])
        reformatted_recons[f'z{j+1}'].append(recons[1][j])
        reformatted_recons[f'z{j+1}'].append(recons[2][j])
        reformatted_recons[f'z{j+1}'].append(recons[3][j])

    for r in range(8):
        gif_im(reformatted_recons[f'z{j+1}'], gts, r, 'image')

    generate_gif('image')


def gif_im(gen_ims, gts, index, type):
    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(wspace=0, hspace=0.05)
    num_rows = 2
    num_cols = 4

    im_er, ax_er = None, None
    for i in range(num_cols):
        generate_image(fig, gts[i], gen_ims[i], '', i + 1, num_rows, num_cols)
        im_er, ax_er = generate_error_map(fig, gts[i], gen_ims[i], i + 5, num_rows, num_cols)

    get_colorbar(fig, im_er, ax_er)
    plt.title(f'Z - {index + 1}', size=8)
    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{index}.png')


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{i}.png'))

    iio.mimsave(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/inpaint_plots/variation_gif.gif', images,
                duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_inpaint.png')


def main(generator, dev_loader):
    for i, data in enumerate(dev_loader):
        input, target_full, mean, std = data

        input = input.to(args.device, dtype=torch.float)
        target = target_full.to(args.device, dtype=torch.float)

        with torch.no_grad():
            recons, gts = create_mean_error_plots(generator, input, target, mean, std)
            # create_z_compare_plots(generator, input, target)
            generate_gifs(recons, gts)

        if i + 1 == 1:
            exit()


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.FloatTensor

    args = create_arg_parser().parse_args()
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
    args.in_chans = 1
    args.out_chans = 1
    _, loader = create_data_loaders(args, val_only=True)
    gen = GANS(args)
    main(gen, loader)
