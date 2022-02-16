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
from utils.training.prepare_data_ablation import create_data_loaders
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
        self.gens = {
            'gens': [],
            'dc': [],
        }
        for i in range(6):
            num = i + 1
            self.gens['gens'].append(self.get_gen(args, num))
            self.gens['dc'].append(True if num == 3 or num == 4 or num == 6 else False)

    def get_gen(self, args, num, type='image'):
        checkpoint_file_gen = pathlib.Path(
            f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/ablation/{type}/{num}/generator_best_model.pt')
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

    def __call__(self, y, mean, std):
        recons = {
            'g1': [],
            'g2': [],
            'g3': [],
            'g4': [],
            'g5': [],
            'g6': [],
        }

        avg = {
            'g1': None,
            'g2': None,
            'g3': None,
            'g4': None,
            'g5': None,
            'g6': None,
        }

        std_devs = {
            'g1': None,
            'g2': None,
            'g3': None,
            'g4': None,
            'g5': None,
            'g6': None,
        }

        batch_size = y.size(0)
        for i in range(len(self.gens['gens'])):
            gen_num = i + 1
            avg_tensor = torch.zeros(8, 16, 128, 128).to(self.args.device)
            for j in range(8):
                z = self.get_noise(batch_size)
                samples = self.gens['gens'][i](y, z)
                samples = readd_measures_im(samples, y, args, true_measures=y) if self.gens['dc'][i] else samples
                avg_tensor[j, :, :, :] = samples[2, :, :, :]

                temp = torch.zeros(8, 128, 128, 2).to(self.args.device)
                temp[:, :, :, 0] = samples[2, 0:8, :, :]
                temp[:, :, :, 1] = samples[2, 8:16, :, :]
                final_im = transforms.root_sum_of_squares(complex_abs(temp * std[2] + mean[2])).cpu().numpy()

                recons[f'g{gen_num}'].append(final_im)

            mean_recon = torch.mean(avg_tensor, dim=0)
            temp = torch.zeros(8, 128, 128, 2).to(self.args.device)
            temp[:, :, :, 0] = mean_recon[0:8, :, :]
            temp[:, :, :, 1] = mean_recon[8:16, :, :]
            avg[f'g{gen_num}'] = transforms.root_sum_of_squares(complex_abs(temp * std[2] + mean[2])).cpu().numpy()

            std_devs[f'g{gen_num}'] = self.compute_std_dev(recons[f'g{gen_num}'], avg[f'g{gen_num}'])

        return recons, avg, std_devs


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
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10]) if not left else fig.add_axes([x10 - 2*pad, y10, width, y11 - y10])

    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2e')  # Generate colorbar
    cbar.ax.tick_params(labelsize=8)

    if left:
        cbar.ax.yaxis.set_label_position('left')


def create_mean_error_plots(avg, std_devs, gt):
    num_rows = 3
    num_cols = 7

    fig = plt.figure(figsize=(5*2.33, 5))
    fig.subplots_adjust(wspace=0, hspace=0.05)
    generate_image(fig, gt, gt, 'GT', 1, num_rows, num_cols)

    labels = ['Adv. Only', '+Supervised', '+DC', '+Var Loss', '+DI - No DC', '+DI - w/ DC']
    im_er, ax_er = None, None
    im_std, ax_std = None, None

    for i in range(num_cols - 1):
        generate_image(fig, gt, avg[f'g{i + 1}'], labels[i], i + 2, num_rows, num_cols)
        if i == 0:
            im_er, ax_er = generate_error_map(fig, gt, avg[f'g{i + 1}'], i + 9, num_rows, num_cols)
            im_std, ax_std = generate_image(fig, gt, std_devs[f'g{i + 1}'], 'Std. Dev', i + 16, num_rows, num_cols)
        else:
            generate_error_map(fig, gt, avg[f'g{i + 1}'], i + 9, num_rows, num_cols)
            generate_image(fig, gt, std_devs[f'g{i + 1}'], 'Std. Dev', i + 16, num_rows, num_cols)

    get_colorbar(fig, im_er, ax_er, left=True)
    get_colorbar(fig, im_std, ax_std, left=True)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/ablation_plots/mean_error.png')


def create_z_compare_plots(recons, gt):
    num_rows = 5
    num_cols = 7

    fig = plt.figure(figsize=(7*1.4, 7))
    fig.subplots_adjust(wspace=0, hspace=0.05)
    generate_image(fig, gt, gt, 'GT', 1, num_rows, num_cols)

    labels = ['Adv. Only', '+Supervised', '+DC', '+Var Loss', '+DI - No DC', '+DI - w/ DC']

    for i in range(num_cols - 1):
        generate_error_map(fig, gt, recons[f'g{i + 1}'][0], i + 2, num_rows, num_cols, title=labels[i])
        generate_error_map(fig, gt, recons[f'g{i + 1}'][1], i + 9, num_rows, num_cols)
        generate_error_map(fig, gt, recons[f'g{i + 1}'][2], i + 16, num_rows, num_cols)
        generate_error_map(fig, gt, recons[f'g{i + 1}'][3], i + 23, num_rows, num_cols)
        generate_error_map(fig, gt, recons[f'g{i + 1}'][4], i + 30, num_rows, num_cols)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/ablation_plots/5_z.png')


def gif_im(gt, gen_ims, index, type):
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(wspace=0, hspace=0.05)
    num_rows = 2
    num_cols = 6

    labels = ['Adv. Only', '+Supervised', '+DC', '+Var Loss', '+DI - No DC', '+DI - w/ DC']

    im_er, ax_er = None, None
    for i in range(num_cols):
        generate_image(fig, gt, gen_ims[f'g{i + 1}'][index], labels[i], i + 1, num_rows, num_cols)
        im_er, ax_er = generate_error_map(fig, gt, gen_ims[f'g{i + 1}'][index], i + 7, num_rows, num_cols)

    get_colorbar(fig, im_er, ax_er)
    plt.title(f'Z - {index + 1}', size=8)
    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{index}.png')


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{i}.png'))

    iio.mimsave(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/ablation_plots/variation_gif.gif', images,
                duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{i}.png')


def main(generators, dev_loader):
    for i, data in enumerate(dev_loader):
        input, target_full, mean, std, true_measures = data

        input = prep_input_2_chan(input, args.network_input, args).to(args.device)
        target_full = prep_input_2_chan(target_full, args.network_input, args).to(args.device)

        temp = torch.zeros(8, 128, 128, 2).to(args.device)
        temp[:, :, :, 0] = target_full[2, 0:8, :, :]
        temp[:, :, :, 1] = target_full[2, 8:16, :, :]
        gt = transforms.root_sum_of_squares(complex_abs(temp * std[2] + mean[2])).cpu().numpy()

        with torch.no_grad():
            recons, avg, std_devs = generators(input, mean, std)

            create_mean_error_plots(avg, std_devs, gt)
            create_z_compare_plots(recons, gt)

            for r in range(8):
                gif_im(gt, recons, r, 'image')

            generate_gif('image')

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
    args.in_chans = 16
    args.out_chans = 16
    _, loader = create_data_loaders(args, val_only=True)
    gens = GANS(args)
    main(gens, loader)
