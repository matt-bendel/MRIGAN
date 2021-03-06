import time
import pickle
import random
import os
import torch
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start, build_model, build_unet
from utils.general.helper import readd_measures_im, prep_input_2_chan
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def get_snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def get_ssim(
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


def non_average_gen(generator, input_w_z, z, old_input):
    start = time.perf_counter()
    output_gen = generator(input=input_w_z, z=z)
    finish = time.perf_counter() - start

    if args.network_input == 'kspace':
        # refined_out = output_gen + old_input[:, 0:16]
        refined_out = output_gen + old_input[:]
    else:
        refined_out = readd_measures_im(output_gen, old_input, args)

    return refined_out


def average_gen(generator, input_w_z, z, old_input):
    average_gen = torch.zeros(input_w_z.shape).to(args.device)

    for j in range(8):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(
            args.device)
        output_gen = generator(input=input_w_z, z=z)

        if args.network_input == 'kspace':
            # refined_out = output_gen + old_input[:, 0:16]
            refined_out = output_gen + old_input[:]
        else:
            refined_out = readd_measures_im(output_gen, old_input, args)

        average_gen = torch.add(average_gen, refined_out)

    return torch.div(average_gen, 8)


def generate_image(fig, target, image, method, image_ind):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT':
        psnr_val = get_psnr(target, image)
        snr_val = get_snr(target, image)
        ssim_val = get_ssim(target, image)
        ax.set_title(f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}')
    ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method}')


def generate_error_map(fig, target, recon, method, image_ind, relative=False, k=1):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmin=0, vmax=0.0001)  # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def get_colorbar(fig, im, ax):
    fig.subplots_adjust(right=0.85)  # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.02
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

    fig.colorbar(im, cax=cbar_ax)  # Generate colorbar


def get_gen(args, type, loc):
    checkpoint_file_gen = pathlib.Path(f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/{type}/{loc}_presentation_temp/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    return generator


def get_unet(args, type):
    checkpoint_file_unet = pathlib.Path(f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/baseline/{type}_best/best_model.pt')
    checkpoint_unet = torch.load(checkpoint_file_unet, map_location=torch.device('cuda'))

    unet = build_unet(args)

    if args.data_parallel:
        unet = torch.nn.DataParallel(unet)

    unet.load_state_dict(checkpoint_unet['model'])

    return unet


def main(args):
    args.in_chans = 2
    args.out_chans = 2

    image_gen_0001 = get_gen(args, 'image', 2)
    image_gen_0001.eval()
    image_gen_001 = get_gen(args, 'image', 3)
    image_gen_001.eval()
    image_gen_01 = get_gen(args, 'image', 1)
    image_gen_01.eval()
    image_unet = get_unet(args, 'image')
    image_unet.eval()

    train_loader, dev_loader = create_data_loaders(args, val_only=True)

    for i, data in enumerate(dev_loader):
        input, target_full, mean, std, nnz_index_mask = data
        input_im = prep_input_2_chan(input, args.network_input, args).to(args.device)
        target_full = prep_input_2_chan(target_full, args.network_input, args)

        with torch.no_grad():
            z = torch.FloatTensor(np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(1))).to(
                args.device)
            image_gen_01 = non_average_gen(image_gen_01, input_im, z, input_im)
            image_gen_001 = non_average_gen(image_gen_001, input_im, z, input_im)
            image_gen_0001 = non_average_gen(image_gen_0001, input_im, z, input_im)
            image_unet_out = image_unet(input_im)

            target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device)
            image_gen_01_batch = prep_input_2_chan(image_gen_01, 'image', args, disc=True).to(args.device)
            image_gen_001_batch = prep_input_2_chan(image_gen_001, 'image', args, disc=True).to(args.device)
            image_gen_0001_batch = prep_input_2_chan(image_gen_0001, 'image', args, disc=True).to(args.device)
            image_unet_batch = prep_input_2_chan(image_unet_out, 'image', args, disc=True).to(args.device)

            for j in range(target_batch.shape[0]):
                if j == 2:
                    true_im = complex_abs(target_batch[j].permute(1, 2, 0))
                    zfr_im = complex_abs(input_im[j].permute(1, 2, 0))
                    gen_image_im_01 = complex_abs(image_gen_01_batch[j].permute(1, 2, 0))
                    gen_image_im_001 = complex_abs(image_gen_001_batch[j].permute(1, 2, 0))
                    gen_image_im_0001 = complex_abs(image_gen_0001_batch[j].permute(1, 2, 0))
                    unet_image_im = complex_abs(image_unet_batch[j].permute(1, 2, 0))

                    true_im_np = true_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    zfr_im_np = zfr_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    gen_image_im_01_np = gen_image_im_01.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    gen_image_im_001_np = gen_image_im_001.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    gen_image_im_0001_np = gen_image_im_0001.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    unet_image_im_np = unet_image_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()

                    fig = plt.figure(figsize=(18, 9))
                    fig.suptitle('Reconstructions')

                    generate_image(fig, true_im_np, true_im_np, 'GT', 1)
                    generate_image(fig, true_im_np, zfr_im_np, 'ZFR', 2)
                    generate_image(fig, true_im_np, gen_image_im_01_np, 'Image Generator (0.01)', 3)
                    generate_image(fig, true_im_np, gen_image_im_001_np, 'Image Generator (0.001)', 4)
                    generate_image(fig, true_im_np, gen_image_im_0001_np, 'Image Generator (0.0001)', 5)
                    generate_image(fig, true_im_np, unet_image_im_np, 'Image U-Net', 6)

                    generate_error_map(fig, true_im_np, zfr_im_np, 'ZFR', 8)
                    generate_error_map(fig, true_im_np, gen_image_im_01_np, 'Image Generator ', 9)
                    generate_error_map(fig, true_im_np, gen_image_im_001_np, 'Image Generator ', 10)
                    generate_error_map(fig, true_im_np, gen_image_im_0001_np, 'Image Generator ', 11)
                    im, ax = generate_error_map(fig, true_im_np, unet_image_im_np, 'Image U-Net', 12)

                    get_colorbar(fig, im, ax)
                    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/recons_{i}.png')

        if i + 1 == 1:
            exit()


if __name__ == '__main__':
    rows = 2
    cols = 6
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

    main(args)
    # plot_loss()
