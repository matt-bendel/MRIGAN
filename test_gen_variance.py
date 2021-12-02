import time
import pickle
import random
import os
import torch
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

from typing import Optional
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
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


def z_gen(generator, input_w_z, z, old_input):
    start = time.perf_counter()
    output_gen = generator(input=input_w_z, z=z)

    if args.network_input == 'kspace':
        # refined_out = output_gen + old_input[:, 0:16]
        refined_out = output_gen + old_input[:]
    else:
        refined_out = readd_measures_im(output_gen, old_input, args)

    return refined_out


def average_gen(generator, input_w_z, z, old_input, args, num_z=8):
    average_gen = torch.zeros(input_w_z.shape).to(args.device)
    average_gen_kspace = torch.zeros(input_w_z.shape).to(args.device)
    gen_list = []
    gen_list_kspace = []
    for j in range(num_z):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(
            args.device)
        output_gen = generator(input=input_w_z, z=z)

        if args.network_input == 'kspace':
            # refined_out = output_gen + old_input[:, 0:16]
            refined_out = output_gen + old_input[:]
        else:
            refined_out = readd_measures_im(output_gen, old_input, args)
            kspace_refined_out = readd_measures_im(output_gen, old_input, args, kspace=True)

        gen_list.append(refined_out)
        gen_list_kspace.append(kspace_refined_out)
        average_gen = torch.add(average_gen, refined_out)
        average_gen_kspace = torch.add(average_gen_kspace, kspace_refined_out)

    return torch.div(average_gen, num_z), gen_list, torch.div(average_gen_kspace, num_z), gen_list_kspace


def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            ax.set_title(f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}')

    if method == 'Std. Dev':
        if kspace:
            image = image**0.45
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image**0.45
            target = target**0.45
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(method)

    return im, ax


def generate_error_map(fig, target, recon, method, image_ind, rows, cols, relative=False, k=1, kspace=False):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    if kspace:
        recon = recon ** 0.45
        target = target ** 0.45

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet') if kspace else ax.imshow(k * error, cmap='jet', vmax=0.0001)

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


def get_gen(args, type):
    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/{type}/{args.z_location}/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    return generator


def get_gen_supervised(args, type):
    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/{type}/2_presentation_temp/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    return generator


def gif_im(true, gen_im, index, type):
    fig = plt.figure()
    if type == 'kspace':
        generate_image(fig, true, gen_im, f'z {index}', 1, 2, 1, kspace=True)
        im, ax = generate_error_map(fig, true, gen_im, f'z {index}', 2, 2, 1, kspace=True)
    else:
        generate_image(fig, true, gen_im, f'z {index}', 1, 2, 1)
        im, ax = generate_error_map(fig, true, gen_im, f'z {index}', 2, 2, 1)

    get_colorbar(fig, im, ax)
    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/gifs/gif_{type}_{index - 1}.png')


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/MRIGAN/gifs/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_{type}_gif.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/MRIGAN/gifs/gif_{type}_{i}.png')


def main(args):
    args.in_chans = 2
    args.out_chans = 2

    gen = get_gen(args, args.network_input)
    best_gen = get_gen_supervised(args, args.network_input)
    gen.eval()
    best_gen.eval()

    train_loader, dev_loader = create_data_loaders(args, val_only=True)

    for i, data in enumerate(dev_loader):
        input, target_full, mean_val, std, nnz_index_mask = data
        kspace_gt = prep_input_2_chan(target_full, 'kspace', args)
        kspace_us = prep_input_2_chan(input, 'kspace', args)
        input = prep_input_2_chan(input, args.network_input, args)
        target_full = prep_input_2_chan(target_full, args.network_input, args)
        old_input = input.to(args.device)

        with torch.no_grad():
            input_w_z = input.to(args.device)
            mean, gens, kspace_mean_batch, kspace_gens = average_gen(gen, input_w_z, None, old_input, args)
            best_mean, best_gens, best_kspace_mean_batch , best_kspace_gens = average_gen(best_gen, input_w_z, None, old_input, args)
            zero = z_gen(gen, input_w_z, torch.zeros((input.shape[0], args.latent_size)), old_input)

            target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device)
            mean_batch = prep_input_2_chan(mean, args.network_input, args, disc=True).to(args.device)
            best_mean_batch = prep_input_2_chan(best_mean, args.network_input, args, disc=True).to(args.device)
            zero_batch = prep_input_2_chan(zero, args.network_input, args, disc=True).to(args.device)
            gens_batch_list = []
            for val in gens:
                gens_batch_list.append(prep_input_2_chan(val, args.network_input, args, disc=True).to(args.device))

            for j in range(mean_batch.shape[0]):
                if j == 7:
                    true_im = complex_abs(target_batch[j].permute(1, 2, 0))
                    kspace_true_mag_np = complex_abs(kspace_gt[j].permute(1, 2, 0)).cpu().numpy()
                    kspace_us_mag_np = complex_abs(kspace_us[j].permute(1, 2, 0)).cpu().numpy()
                    gen_mean_im = complex_abs(mean_batch[j].permute(1, 2, 0))
                    kspace_mean_mag_np = complex_abs(kspace_mean_batch[j].permute(1, 2, 0)).cpu().numpy()
                    best_gen_mean_im = complex_abs(best_mean_batch[j].permute(1, 2, 0))
                    best_kspace_mean_mag = complex_abs(best_kspace_mean_batch[j].permute(1, 2, 0)).cpu().numpy()
                    zero_im = complex_abs(zero_batch[j].permute(1, 2, 0))
                    gens_im_list = []
                    for val in gens_batch_list:
                        gens_im_list.append(complex_abs(val[j].permute(1, 2, 0)))

                    true_im_np = true_im.cpu().numpy() * std[j].cpu().numpy() + mean_val[j].cpu().numpy()
                    gen_mean_im_np = gen_mean_im.cpu().numpy() * std[j].cpu().numpy() + mean_val[j].cpu().numpy()
                    best_gen_mean_im_np = best_gen_mean_im.cpu().numpy() * std[j].cpu().numpy() + mean_val[
                        j].cpu().numpy()
                    zero_im_np = zero_im.cpu().numpy() * std[j].cpu().numpy() + mean_val[j].cpu().numpy()

                    gen_im_np_list = []
                    for val in gens_im_list:
                        gen_im_np_list.append(val.cpu().numpy() * std[j].cpu().numpy() + mean_val[j].cpu().numpy())

                    gen_kspace_np_list = []
                    for val in kspace_gens:
                        gen_kspace_np_list.append(complex_abs(val[j].permute(1, 2, 0)).cpu().numpy())

                    std_dev = np.zeros(gen_mean_im_np.shape)
                    for val in gen_im_np_list:
                        std_dev = std_dev + np.power((val - gen_mean_im_np), 2)

                    std_dev = std_dev / 8
                    std_dev = np.sqrt(std_dev)

                    std_dev_kspace = np.zeros(kspace_mean_mag_np.shape)
                    for val in gen_im_np_list:
                        std_dev = std_dev + np.power((val - kspace_mean_mag_np), 2)

                    std_dev_kspace = std_dev_kspace / 8
                    std_dev_kspace = np.sqrt(std_dev_kspace)

                    fig = plt.figure()
                    generate_image(fig, true_im_np, true_im_np, 'GT', 1, 2, 2)
                    generate_image(fig, true_im_np, zero_im_np, 'Z=0', 2, 2, 2)
                    im, ax = generate_error_map(fig, true_im_np, zero_im_np, f'Error', 4, 2, 2)
                    get_colorbar(fig, im, ax)
                    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/z_0_{args.network_input}.png')

                    fig = plt.figure(figsize=(18, 9))

                    generate_image(fig, true_im_np, true_im_np, 'GT', 1, 2, 4)
                    generate_image(fig, true_im_np, best_gen_mean_im_np, 'Supervised', 2, 2, 4)
                    generate_image(fig, true_im_np, gen_mean_im_np, 'Mean', 3, 2, 4)
                    im, ax = generate_image(fig, true_im_np, std_dev, 'Std. Dev', 4, 2, 4)
                    get_colorbar(fig, im, ax)
                    generate_error_map(fig, true_im_np, best_gen_mean_im_np, f'Error', 6, 2, 4)
                    im, ax = generate_error_map(fig, true_im_np, gen_mean_im_np, f'Error', 7, 2, 4)
                    get_colorbar(fig, im, ax)

                    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/mean_and_std_{args.network_input}.png')

                    fig = plt.figure(figsize=(18, 9))

                    generate_image(fig, kspace_true_mag_np, kspace_true_mag_np, 'GT', 1, 2, 5, kspace=True)
                    generate_image(fig, kspace_true_mag_np, kspace_us_mag_np, 'Undersampled', 2, 2, 5, kspace=True)
                    generate_image(fig, kspace_true_mag_np, best_kspace_mean_mag, 'Supervised', 3, 2, 5, kspace=True)
                    generate_image(fig, kspace_true_mag_np, kspace_mean_mag_np, 'Mean', 4, 2, 5, kspace=True)
                    im, ax = generate_image(fig, kspace_true_mag_np, std_dev_kspace, 'Std. Dev', 5, 2, 5)
                    get_colorbar(fig, im, ax)

                    generate_error_map(fig, kspace_true_mag_np, kspace_us_mag_np, f'Error', 7, 2, 5, kspace=True)
                    generate_error_map(fig, kspace_true_mag_np, best_kspace_mean_mag, f'Error', 8, 2, 5, kspace=True)
                    im, ax = generate_error_map(fig, kspace_true_mag_np, kspace_mean_mag_np, f'Error', 9, 2, 5, kspace=True)
                    get_colorbar(fig, im, ax)

                    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/mean_and_std_kspace.png')

                    fig = plt.figure(figsize=(14, 14))
                    place = 1
                    for val in gen_im_np_list:
                        if place <= 4:
                            generate_image(fig, true_im_np, val, f'z {place}', place, 4, 4)
                            im, ax = generate_error_map(fig, true_im_np, val, f'z {place}', place + 4, 4, 4)
                        else:
                            generate_image(fig, true_im_np, val, f'z {place}', place + 4, 4, 4)
                            im, ax = generate_error_map(fig, true_im_np, val, f'z {place}', place + 8, 4, 4)
                        place += 1
                        if place > 8:
                            break

                    get_colorbar(fig, im, ax)
                    plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/comparison_{args.network_input}.png')

                    #TODO: GENERATE KSPACE GIF
                    place = 1
                    for val in gen_im_np_list:
                        gif_im(true_im_np, val, place, 'image')
                        place += 1

                    generate_gif('image')

                    place = 1
                    for val in gen_kspace_np_list:
                        gif_im(kspace_true_mag_np, val, place, 'kspace')
                        place += 1

                    generate_gif('kspace')

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

    main(args)
    # plot_loss()
