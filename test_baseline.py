import time
import pickle
import random
import os
import torch
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from data import transforms
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train_unet
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


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 2
    args.out_chans = 2
    args.checkpoint = pathlib.Path(f'/home/bendel.8/Git_Repos/MRIGAN/trained_models/baseline/{args.network_input}/best_model.pt')
    unet, opt, args, best_dev_loss, start_epoch = resume_train_unet(args)
    unet.eval()
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

            with torch.no_grad():
                input_w_z = input.to(args.device)
                start = time.perf_counter()
                refined_out = unet(input_w_z)
                finish = time.perf_counter() - start

                target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device)
                output_batch = prep_input_2_chan(refined_out, args.network_input, args, disc=True).to(args.device)

                metrics['time'] = finish / output_batch.shape[0]

                batch_metrics = {
                    'psnr': [],
                    'ssim': [],
                    'snr': []
                }


                for j in range(output_batch.shape[0]):
                    output_rss = torch.zeros(8, output_batch.shape[2], output_batch.shape[2], 2).to(args.device)
                    output_rss[:, :, :, 0] = output_batch[j, 0:8, :, :]
                    output_rss[:, :, :, 1] = output_batch[j, 8:16, :, :]
                    generared_im = transforms.root_sum_of_squares(complex_abs(output_rss * std[i] + mean[i]))

                    target_rss = torch.zeros(8, target_batch.shape[2], target_batch.shape[2], 2).to(args.device)
                    target_rss[:, :, :, 0] = target_batch[j, 0:8, :, :]
                    target_rss[:, :, :, 1] = target_batch[j, 8:16, :, :]
                    true_im = transforms.root_sum_of_squares(complex_abs(target_rss * std[i] + mean[i]))

                    generated_im_np = generared_im.cpu().numpy()
                    true_im_np = true_im.cpu().numpy()

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

        print(np.median(metrics['snr']))
        print(np.median(metrics['psnr']))
        print(np.median(metrics['ssim']))

        save_str = f"[Avg. PSNR: {np.mean(metrics['psnr'])}] [Avg. SNR: {np.mean(metrics['snr'])}] [Avg. SSIM: {np.mean(metrics['ssim'])}], [Avg. Time: {np.mean(metrics['time'])}]"
        metric_file.write(save_str)
        print(save_str)



if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.FloatTensor

    args = create_arg_parser().parse_args()
    args.batch_size = 8
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
