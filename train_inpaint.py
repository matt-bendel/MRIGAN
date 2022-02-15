"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Adapted by: Matt Bendel based on work originally by Saurav

SPECIFICATIONS FOR TRAINING:
- Brain data onlu.
  - Can select contrast.
- GRO sampling pattern with R=4.
- All multicoil data where num_coils > 8 was condensed to 8 coil
- Can either train k-space U-Net or image space U-Net
- Base U-Net in either case has 16 input channels:
  - 8 per coil for real values
  - 8 per coil for complex values
"""
import pathlib
import logging
import pickle
import random
import os
import shutil
import torch
# import pytorch_ssim
import pytorch_msssim

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt
import imageio as iio

from typing import Optional
from data import transforms
from utils.math import complex_abs
from utils.training.prepare_data_inpaint import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start
from utils.general.helper import readd_measures_im, prep_input_2_chan
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Tunable weight for gradient penalty
lambda_gp = 10

GLOBAL_LOSS_DICT = {
    'g_loss': [],
    'd_loss': [],
    'mSSIM': [],
    'd_acc': []
}

CONSTANT_PLOTS = {
    'measures': None,
    'mean': None,
    'std': None,
    'gt': None
}


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


def mssim_tensor(gt, pred, epoch):
    """ Compute Normalized Mean Squared Error (NMSE) """
    # ssim_loss = pytorch_ssim.SSIM()
    return pytorch_msssim.msssim(gt, pred, normalize=True) if epoch == 1 else pytorch_msssim.msssim(gt, pred)


def save_model(args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': args.exp_dir
        },
        f=args.exp_dir / 'inpaint' / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(args.exp_dir / 'inpaint' / f'{m_type}_model.pt',
                        args.exp_dir / 'inpaint' / f'{m_type}_best_model.pt'
                        )


def compute_gradient_penalty(D, real_samples, fake_samples, args, y):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input=interpolates, y=y)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(args.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def average_gen(generator, input_w_z, args, target=None, inds=None):
    gens = torch.zeros(size=(input_w_z.shape[0], 8, input_w_z.shape[1], input_w_z.shape[2], input_w_z.shape[3])).to(device=args.device, dtype=torch.float)
    for j in range(8):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(
            device=args.device, dtype=torch.float)
        output_gen = generator(input=input_w_z, z=z)
        output_gen[inds] = target[inds]
        gens[:, j, :, :, :] = output_gen

    return torch.mean(gens, dim=1), gens


def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            pred = disc_num
            ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}, Pred: {pred * 100:.2f}% True') if disc_num else ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
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
        recon = recon ** 0.4
        target = target ** 0.4

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def gif_im(true, gen_im, index, type, disc_num=False):
    fig = plt.figure()

    generate_image(fig, true, gen_im, f'z {index}', 1, 2, 1, disc_num=False)
    im, ax = generate_error_map(fig, true, gen_im, f'z {index}', 2, 2, 1)

    plt.savefig(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{index - 1}.png')
    plt.close()


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_{type}_gif.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/gifs/gif_{type}_{i}.png')


def get_gen_supervised(args):
    from utils.training.prepare_model import build_model, build_optim, build_discriminator

    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/inpaint/ours/generator_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    checkpoint_file_dis = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/inpaint/ours/discriminator_model.pt')
    checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

    generator = build_model(args)
    discriminator = build_discriminator(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    generator.load_state_dict(checkpoint_gen['model'])

    opt_gen = build_optim(args, generator.parameters())
    opt_gen.load_state_dict(checkpoint_gen['optimizer'])

    discriminator.load_state_dict(checkpoint_dis['model'])

    opt_dis = build_optim(args, discriminator.parameters())
    opt_dis.load_state_dict(checkpoint_dis['optimizer'])

    return generator, opt_gen, discriminator, opt_dis, args, checkpoint_gen['best_dev_loss'], checkpoint_dis[
        'best_dev_loss'], checkpoint_gen['epoch']


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 1
    args.out_chans = 1
    # args.batch_size = 32

    if args.resume:
        generator, optimizer_G, discriminator, optimizer_D, args, best_loss_val, best_loss_dis, start_epoch = get_gen_supervised(
            args)
        start_epoch = start_epoch + 1
    else:
        generator, discriminator, best_dev_loss, start_epoch = fresh_start(args)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        best_loss_val = 0

    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader = create_data_loaders(args)

    for epoch in range(start_epoch, args.num_epochs):
        batch_loss = {
            'g_loss': [],
            'd_loss': [],
            'd_acc': []
        }

        for i, data in enumerate(train_loader):
            input, target, mean, std = data

            inds = torch.nonzero(input)

            input = input.to(device=args.device, dtype=torch.float)
            target = target.to(device=args.device, dtype=torch.float)

            for j in range(args.num_iters_discriminator):
                z = torch.FloatTensor(
                    np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(1))).to(device=args.device, dtype=torch.float)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                output_gen = generator(input, z)
                output_gen[inds] = target[inds]

                # MAKE PREDICTIONS
                real_pred = discriminator(input=target, y=input)
                fake_pred = discriminator(input=output_gen, y=input)

                real_acc = real_pred[real_pred > 0].shape[0]
                fake_acc = fake_pred[fake_pred <= 0].shape[0]

                batch_loss['d_acc'].append((real_acc + fake_acc) / (2 * real_pred.shape[0]))

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, target.data,
                                                            output_gen.data, args, input.data)
                # Adversarial loss
                d_loss = torch.mean(fake_pred) - torch.mean(
                    real_pred) + lambda_gp * gradient_penalty + 0.001 * torch.mean(real_pred ** 2)

                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()

            # Generate a batch of images
            z = torch.FloatTensor(
                np.random.normal(size=(args.num_z, input.shape[0], args.latent_size), scale=np.sqrt(1))).to(
                args.device)
            output_gen = torch.zeros(size=(
                args.num_z, input.shape[0], input.shape[1], input.shape[2], input.shape[3])).to(
                args.device)
            for k in range(args.num_z):
                output_gen[k, :, :, :, :] = generator(input, z[k])
                # output_gen[k, :, :, :, 0:64] = target[:, :, :, 0:64]
                output_gen[k, inds] = target[inds]

            disc_inputs_gen = torch.zeros(
                size=(input.shape[0], args.num_z, output_gen.shape[2], output_gen.shape[3],
                      output_gen.shape[4])
            ).to(args.device)
            for l in range(input.shape[0]):
                for k in range(args.num_z):
                    disc_inputs_gen[l, k, :, :, :] = output_gen[k, l, :, :, :]

            avg_recon = torch.mean(disc_inputs_gen, dim=1)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_pred = torch.zeros((input.shape[0], args.num_z)).to(args.device)
            for k in range(input.shape[0]):
                cond = torch.zeros(1, disc_inputs_gen.shape[2], disc_inputs_gen.shape[3], disc_inputs_gen.shape[4])
                cond[0, :, :, :] = input[k, :, :, :]
                cond = cond.repeat(args.num_z, 1, 1, 1)
                temp = discriminator(input=disc_inputs_gen[k], y=cond)
                fake_pred[k] = temp[:, 0]

            gen_pred_loss = torch.mean(fake_pred[0])
            for k in range(input.shape[0] - 1):
                gen_pred_loss += torch.mean(fake_pred[k + 1])

            var_weight = 0.02
            adv_weight = 1e-6
            ssim_weight = 0.84
            g_loss = -adv_weight * torch.mean(gen_pred_loss)
            g_loss += (1 - ssim_weight) * F.l1_loss(target, avg_recon) - ssim_weight * mssim_tensor(target, avg_recon, epoch+1)
            g_loss += - var_weight * torch.mean(torch.var(disc_inputs_gen, dim=1), dim=(0, 1, 2, 3))

            g_loss.backward()
            optimizer_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, args.num_epochs, i, len(train_loader.dataset) / args.batch_size, d_loss.item(),
                   g_loss.item())
            )

        losses = {
            'psnr': [],
            'ssim': []
        }

        for i, data in enumerate(dev_loader):
            generator.eval()
            with torch.no_grad():
                input, target_im, mean, std = data

                inds = torch.nonzero(input==0)

                input = input.to(device=args.device, dtype=torch.float)
                target_im = target_im.to(device=args.device, dtype=torch.float)

                output_gen, gen_list = average_gen(generator, input, args, target=target_im, inds=inds)

                for k in range(input.shape[0]):
                    output = output_gen[k].squeeze(0).cpu().numpy()
                    target = target_im[k].squeeze(0).cpu().numpy()

                    losses['ssim'].append(ssim(target, output))
                    losses['psnr'].append(psnr(target, output))

                    if i + 1 == 1 and k == 2:
                        std_dev = torch.std(gen_list[k], dim=0).squeeze(0).cpu().numpy()

                        place = 1
                        for r in range(8):
                            val = gen_list[k, r, 0, :, :].cpu().numpy()
                            gif_im(target, val, place, 'image')
                            place += 1

                        generate_gif('image')

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        im = ax.imshow(std_dev, cmap='viridis')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

                        fig.colorbar(im, cax=cbar_ax)

                        plt.savefig(f'std_dev_gen_ablation_{args.z_location}.png')
                        plt.close()

                        plt.figure()
                        plt.imshow(np.abs(output), cmap='gray')
                        plt.savefig(f'temp_gen_out_ablation_{args.z_location}.png')
                        plt.close()

                        plt.figure()
                        plt.imshow(np.abs(target), cmap='gray')
                        plt.savefig('temp_gen_targ_ablation.png')
                        plt.close()

                if i == 10:
                    break

        psnr_loss = np.mean(losses['psnr'])
        best_model = psnr_loss > best_loss_val
        best_loss_val = psnr_loss if psnr_loss > best_loss_val else best_loss_val

        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
        GLOBAL_LOSS_DICT['d_acc'].append(np.mean(batch_loss['d_acc']))

        save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch - start_epoch]:.4f}] [Average D Acc: {GLOBAL_LOSS_DICT['d_acc'][epoch - start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch - start_epoch]:.4f}]\n"
        print(save_str)

        save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
        print(save_str_2)

        save_model(args, epoch, generator, optimizer_G, best_loss_val, best_model, 'generator')
        save_model(args, epoch, discriminator, optimizer_D, best_loss_val, best_model, 'discriminator')

        generator.train()


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
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main(args)
