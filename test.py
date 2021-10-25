import time
import pickle
import random
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start
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


def non_average_gen(generator, input_w_z, z, old_input):
    start = time.perf_counter()
    output_gen = generator(input=input_w_z, z=z)
    finish = time.perf_counter() - start

    if args.network_input == 'kspace':
        # refined_out = output_gen + old_input[:, 0:16]
        refined_out = output_gen + old_input[:]
    else:
        refined_out = readd_measures_im(output_gen, old_input, args)

    return refined_out, finish


def average_gen(generator, input_w_z, z, old_input):
    start = time.perf_counter()
    average_gen = torch.zeros(input_w_z.shape).to(args.device)

    for j in range(8):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(args.device)
        output_gen = generator(input=input_w_z, z=z)

        if args.network_input == 'kspace':
            # refined_out = output_gen + old_input[:, 0:16]
            refined_out = output_gen + old_input[:]
        else:
            refined_out = readd_measures_im(output_gen, old_input, args)

        average_gen = torch.add(average_gen, refined_out)

    finish = time.perf_counter() - start

    return torch.div(average_gen, 8), finish


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 17 if args.z_location == 3 else 2
    args.out_chans = 2

    generator, optimizer_G, discriminator, optimizer_D, args, best_dev_loss, start_epoch = resume_train(args)
    generator.eval()
    train_loader, dev_loader = create_data_loaders(args, val_only=True)

    with open(f'trained_models/{args.network_input}/metrics_{args.z_location}.txt', 'w') as metric_file:
        metrics = {
            'ssim': [],
            'psnr': [],
            'snr': [],
            'time': []
        }

        for i, data in enumerate(dev_loader):
            input, target_full, mean, std, nnz_index_mask = data

            input = prep_input_2_chan(input, args.network_input, args)
            target_full = prep_input_2_chan(target_full, args.network_input, args)
            z = torch.FloatTensor(np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(1))).to(args.device)
            old_input = input.to(args.device)

            with torch.no_grad():
                input_w_z = input.to(args.device)
                refined_out, finish = non_average_gen(generator, input_w_z, z, old_input)
                # refined_out, finish = average_gen(generator, input_w_z, z, old_input)

                target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device)
                output_batch = prep_input_2_chan(refined_out, args.network_input, args, disc=True).to(args.device)

                metrics['time'] = finish / output_batch.shape[0]

                batch_metrics = {
                    'psnr': [],
                    'ssim': [],
                    'snr': []
                }


                for j in range(output_batch.shape[0]):
                    generared_im = complex_abs(output_batch[j].permute(1, 2, 0))
                    true_im = complex_abs(target_batch[j].permute(1, 2, 0))

                    generated_im_np = generared_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()
                    true_im_np = true_im.cpu().numpy() * std[j].numpy() + mean[j].numpy()

                    batch_metrics['psnr'].append(psnr(true_im_np, generated_im_np, np.max(true_im_np)))
                    batch_metrics['ssim'].append(ssim(true_im_np, generated_im_np, np.max(true_im_np)))
                    batch_metrics['snr'].append(snr(true_im_np, generated_im_np))

                metrics['psnr'].append(np.mean(batch_metrics['psnr']))
                metrics['snr'].append(np.mean(batch_metrics['snr']))
                metrics['ssim'].append(np.mean(batch_metrics['ssim']))

                print(
                    "[Avg. Batch PSNR %.2f] [Avg. Batch SNR %.2f]  [Avg. Batch SSIM %.4f]"
                    % (np.mean(batch_metrics['psnr']), np.mean(batch_metrics['snr']), np.mean(batch_metrics['ssim']))
                )

        save_str = f"[Avg. PSNR: {np.mean(metrics['psnr'])}] [Avg. SNR: {np.mean(metrics['snr'])}] [Avg. SSIM: {np.mean(metrics['ssim'])}], [Avg. Time: {np.mean(metrics['time'])}]"
        metric_file.write(save_str)
        print(save_str)



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
