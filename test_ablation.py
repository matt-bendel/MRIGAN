import time
import pickle
import random
import os
import torch
import pathlib

import numpy as np
import matplotlib.pyplot as plt

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


def non_average_gen(generator, input_w_z, z, old_input, args, true_measures):
    start = time.perf_counter()
    output_gen = generator(input=input_w_z, z=z)
    finish = time.perf_counter() - start

    if args.network_input == 'kspace':
        # refined_out = output_gen + old_input[:, 0:16]
        refined_out = output_gen + old_input[:]
    else:
        refined_out = readd_measures_im(output_gen, old_input, args,
                                        true_measures=true_measures) if args.data_consistency else output_gen

    return refined_out, finish


def average_gen(generator, input_w_z, z, old_input, args, true_measures, num_code=1024):
    start = time.perf_counter()
    average_gen = torch.zeros((input_w_z.shape[0], num_code, 16, 128, 128)).to(args.device)

    for j in range(num):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(
            args.device)
        output_gen = generator(input=input_w_z, z=z)

        if args.network_input == 'kspace':
            # refined_out = output_gen + old_input[:, 0:16]
            refined_out = output_gen + old_input[:]
        else:
            refined_out = readd_measures_im(output_gen, old_input, args,
                                            true_measures=true_measures) if args.data_consistency else output_gen

        average_gen[:, j, :, :, :] = refined_out

    finish = time.perf_counter() - start

    return torch.mean(average_gen, dim=1), finish, torch.mean(torch.std(average_gen, dim=1), dim=(0, 1, 2, 3))


def get_gen(args, type='image'):
    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/ablation/{type}/{args.z_location}/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    return generator


def main(args, num, network):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 16
    args.out_chans = 16
    args.z_location = network
    if network == 3 or network == 4 or network == 6:
        args.data_consistency = True

    generator = get_gen(args)
    generator.eval()
    train_loader, dev_loader = create_data_loaders(args, val_only=True)

    with open(f'trained_models/{args.network_input}/metrics_{args.z_location}.txt', 'w') as metric_file:
        metrics = {
            'ssim': [],
            'psnr': [],
            'snr': [],
            'apsd': [],
            'time': []
        }
        first = True
        for i, data in enumerate(dev_loader):
            input, target_full, mean, std, true_measures = data

            input = prep_input_2_chan(input, args.network_input, args)
            target_full = prep_input_2_chan(target_full, args.network_input, args)
            z = torch.FloatTensor(np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(1))).to(
                args.device)
            old_input = input.to(args.device)

            with torch.no_grad():
                input_w_z = input.to(args.device)
                # refined_out, finish = non_average_gen(generator, input_w_z, z, old_input)
                refined_out, finish, apsd = average_gen(generator, input_w_z, z, old_input, args, true_measures, num_code=num)

                target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device)
                output_batch = prep_input_2_chan(refined_out, args.network_input, args, disc=True).to(args.device)

                metrics['time'] = finish / output_batch.shape[0]
                metrics['apsd'] = apsd

                batch_metrics = {
                    'psnr': [],
                    'ssim': [],
                    'snr': []
                }

                for j in range(output_batch.shape[0]):
                    output_rss = torch.zeros(8, output_batch.shape[2], output_batch.shape[2], 2)
                    output_rss[:, :, :, 0] = output_batch[j, 0:8, :, :]
                    output_rss[:, :, :, 1] = output_batch[j, 8:16, :, :]
                    output = transforms.root_sum_of_squares(complex_abs(output_rss * std[j] + mean[j]))

                    target_rss = torch.zeros(8, target_batch.shape[2], target_batch.shape[2], 2)
                    target_rss[:, :, :, 0] = target_batch[j, 0:8, :, :]
                    target_rss[:, :, :, 1] = target_batch[j, 8:16, :, :]
                    target = transforms.root_sum_of_squares(complex_abs(target_rss * std[j] + mean[j]))

                    generated_im_np = output.cpu().numpy()
                    true_im_np = target.cpu().numpy()

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

        print(f"RESULTS FOR {num} CODE VECTORS")
        save_str = f"[Avg. PSNR: {np.mean(metrics['psnr']):.2f}] [Avg. SNR: {np.mean(metrics['snr']):.2f}] [Avg. SSIM: {np.mean(metrics['ssim']):.4f}], [Avg. APSD: {np.mean(metrics['apsd'])}], [Avg. Time: {np.mean(metrics['time']):.3f}]"
        metric_file.write(save_str)
        print(f"[Median PSNR {np.median(metrics['psnr']):.2f}")
        print(f"[Median SNR {np.median(metrics['snr']):.2f}")
        print(f"[Median SSIM {np.median(metrics['ssim']):.4f}")
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

    for net in range(6):
        print(f"VALIDATING ABLATION NETWORK {net+1}")
        for num in range(11):
            power = (2**num)//1
            print(f"VALIDATING NUM CODE VECTORS: {power}")
            main(args, power, net+1)
        print("\n\n\n")
