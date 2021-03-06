import time
import pickle
import random
import os
import torch
import pathlib
import math

import numpy as np
import matplotlib.pyplot as plt

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


def average_gen(generator, input_w_z, args, target=None, inds=None, power_num=128):
    start = time.perf_counter()
    average_gen = torch.zeros((input_w_z.shape[0], power_num, 1, 128, 128)).to(args.device)

    for j in range(power_num):
        z = torch.randn((input_w_z.size(0), 2, 128, 128)).cuda()
        output_gen = generator(torch.cat([input_w_z, z]))
        output_gen[inds] = target[inds]
        average_gen[:, j, :, :, :] = output_gen

    finish = time.perf_counter() - start

    return torch.mean(average_gen, dim=1), finish, torch.mean(torch.std(average_gen, dim=1), dim=(0, 1, 2, 3)).cpu().numpy()


def get_gen(args, type='image'):
    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/inpaint/generator_best_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    print(checkpoint_gen['epoch'])

    return generator


def main(args, power_num, generator, dev_loader):
    # generator = get_gen(args)
    # generator.eval()
    # _, dev_loader = create_data_loaders(args, val_only=True)

    metrics = {
        'ssim': [],
        'psnr': [],
        'snr': [],
        'apsd': [],
        'time': []
    }

    for i, data in enumerate(dev_loader):
        input, target, mean, std = data

        inds = torch.nonzero(input == 0)
        input = input.to(args.device, dtype=torch.float)
        target = target.to(args.device, dtype=torch.float)

        with torch.no_grad():
            # refined_out, finish = non_average_gen(generator, input_w_z, z, old_input)
            output, finish, apsd = average_gen(generator, input, args, target=target, inds=inds, power_num=power_num)

            metrics['time'] = finish / target.shape[0]
            apsd = 0 if math.isnan(apsd) else apsd
            metrics['apsd'] = apsd

            batch_metrics = {
                'psnr': [],
                'ssim': [],
                'snr': []
            }

            for j in range(output.shape[0]):
                generated_im_np = output[j].squeeze(0).cpu().numpy() * std[j].numpy() + mean[j].numpy()
                true_im_np = target[j].squeeze(0).cpu().numpy() * std[j].numpy() + mean[j].numpy()

                batch_metrics['psnr'].append(psnr(true_im_np, generated_im_np, np.max(true_im_np)))
                batch_metrics['ssim'].append(ssim(true_im_np, generated_im_np, np.max(true_im_np)))
                batch_metrics['snr'].append(snr(true_im_np, generated_im_np))

            for j in range(len(batch_metrics['psnr'])):
                metrics['psnr'].append(batch_metrics['psnr'][j])
                metrics['snr'].append(batch_metrics['snr'][j])
                metrics['ssim'].append(batch_metrics['ssim'][j])

            # print(
            #     "[Avg. Batch PSNR %.2f] [Avg. Batch SNR %.2f]  [Avg. Batch SSIM %.4f]"
            #     % (np.mean(batch_metrics['psnr']), np.mean(batch_metrics['snr']), np.mean(batch_metrics['ssim']))
            # )

    fold_psnr = []
    fold_ssim = []
    fold_snr = []
    print(len(metrics['psnr']))
    for l in range(26):
        fold_psnr.append(metrics['psnr'][l * 72:(l + 1) * 72])
        fold_snr.append(metrics['snr'][l * 72:(l + 1) * 72])
        fold_ssim.append(metrics['ssim'][l * 72:(l + 1) * 72])

    # save_str = f"[Avg. PSNR: {np.mean(metrics['psnr']):.2f}] [Avg. SNR: {np.mean(metrics['snr']):.2f}] [Avg. SSIM: {np.mean(metrics['ssim']):.4f}], [Avg. APSD: {np.mean(metrics['apsd'])}], [Avg. Time: {np.mean(metrics['time']):.3f}]"
    print(f'PSNR: {np.mean(fold_psnr)} \\pm {np.std(fold_psnr)}')
    print(f'SNR: {np.mean(fold_snr)} \\pm {np.std(fold_snr)}')
    print(f'SSIM: {np.mean(fold_ssim)} \\pm {np.std(fold_ssim)}')
    print(f'APSD: {np.mean(metrics["apsd"])} \\pm {np.std(metrics["apsd"])}')


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.FloatTensor

    args = create_arg_parser().parse_args()
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 1
    args.out_chans = 1
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
    _, loader = create_data_loaders(args, val_only=True, big_test=True)

    gen = get_gen(args)
    gen.eval()

    power = 128
    main(args, power, gen, loader)

    # main(args)
