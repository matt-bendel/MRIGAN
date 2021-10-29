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
import pytorch_ssim

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt

from typing import Optional
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start
from utils.general.helper import get_inverse_mask, readd_measures_im, prep_input_2_chan
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


def mssim_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    ssim_loss = pytorch_ssim.SSIM()
    return ssim_loss(gt, pred)


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
        f=args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_model.pt',
                        args.exp_dir / args.network_input / str(args.z_location) / f'{m_type}_best_model.pt'
                        )


def compute_gradient_penalty(D, real_samples, fake_samples, args):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
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


def generate_image(fig, target, image, title, image_ind):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(2, 3, image_ind)
    ax.set_title(title)
    ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
    ax.set_xticks([])
    ax.set_yticks([])


def generate_error_map(fig, target, recon, image_ind, k=5, max=1):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(2, 3, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    error = np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    im = ax.imshow(k * error, cmap='jet', vmax=max)  # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def plot_epoch(args, generator, epoch):
    std = CONSTANT_PLOTS['std']
    mean = CONSTANT_PLOTS['mean']

    z_1 = CONSTANT_PLOTS['measures'].unsqueeze(0).to(args.device)
    z = torch.FloatTensor(np.random.normal(size=(z_1.shape[0], args.latent_size))).to(args.device)

    generator.eval()
    with torch.no_grad():
        z_1_out = generator(input=z_1, z=z)

    if args.network_input == 'kspace':
        refined_z_1_out = z_1_out.cpu() + CONSTANT_PLOTS['measures'].unsqueeze(0)
    else:
        refined_z_1_out = readd_measures_im(z_1_out.cpu(), CONSTANT_PLOTS['measures'].unsqueeze(0), args)

    target_prep = prep_input_2_chan(CONSTANT_PLOTS['gt'].unsqueeze(0), args.network_input, args, disc=True)[0]
    zfr = prep_input_2_chan(CONSTANT_PLOTS['measures'].unsqueeze(0), args.network_input, args, disc=True)[0]
    z_1_prep = prep_input_2_chan(refined_z_1_out, args.network_input, args, disc=True)[0]

    target_im = complex_abs(target_prep.permute(1, 2, 0)) * std + mean
    target_im = target_im.cpu().numpy()

    zfr = complex_abs(zfr.permute(1, 2, 0)) * std + mean
    zfr = zfr.cpu().numpy()

    z_1_im = complex_abs(z_1_prep.permute(1, 2, 0)) * std + mean
    z_1_im = z_1_im.detach().cpu().numpy()

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(f'Generated and GT Images at Epoch {epoch + 1}')
    generate_image(fig, target_im, target_im, 'GT', 1)
    generate_image(fig, target_im, zfr, 'ZFR', 2)
    generate_image(fig, target_im, z_1_im, 'Z 1', 3)

    max_val = np.max(np.abs(target_im - zfr))
    generate_error_map(fig, target_im, zfr, 5, 1, max_val)
    generate_error_map(fig, target_im, z_1_im, 6, 1, max_val)

    plt.savefig(
        f'/home/bendel.8/Git_Repos/MRIGAN/training_images/2_chan_z_mid/gen_{args.network_input}_{args.z_location}_{epoch + 1}.png')


def save_metrics(args):
    with open(f'/home/bendel.8/Git_Repos/MRIGAN/saved_metrics/loss_{args.network_input}_{args.z_location}.pkl',
              'wb') as f:
        pickle.dump(GLOBAL_LOSS_DICT, f, pickle.HIGHEST_PROTOCOL)


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 17 if args.z_location == 3 else 2
    args.out_chans = 2

    mse = torch.nn.MSELoss()

    if args.resume:
        generator, optimizer_G, discriminator, optimizer_D, args, best_dev_loss, start_epoch = resume_train(args)
    else:
        generator, discriminator, best_dev_loss, start_epoch = fresh_start(args)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))

    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader = create_data_loaders(args)

    first = True
    best_loss_val = 0

    with open(f'trained_models/{args.network_input}/loss_{args.z_location}.txt', 'w') as loss_file:
        for epoch in range(start_epoch, args.num_epochs):
            batch_loss = {
                'g_loss': [],
                'd_loss': [],
                'd_acc': []
            }

            for i, data in enumerate(train_loader):
                input, target_full, mean, std, nnz_index_mask = data

                input = prep_input_2_chan(input, args.network_input, args)
                target_full = prep_input_2_chan(target_full, args.network_input, args)

                old_input = input.to(args.device)

                input_w_z = input  # add_z_to_input(args, input)

                for j in range(args.num_iters_discriminator):
                    z = torch.FloatTensor(
                        np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(0.001))).to(args.device)
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    optimizer_D.zero_grad()

                    input_w_z = input_w_z.to(args.device)
                    output_gen = generator(input_w_z, z)
                    if args.network_input == 'kspace':
                        # refined_out = output_gen + old_input[:, 0:16]
                        refined_out = output_gen + old_input[:]
                    else:
                        refined_out = readd_measures_im(output_gen, old_input, args)

                    # TURN OUTPUT INTO IMAGE FOR DISCRIMINATION AND GET REAL IMAGES FOR DISCRIMINATION
                    disc_target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True,
                                                          disc_image=not args.disc_kspace).to(
                        args.device)
                    disc_output_batch = prep_input_2_chan(refined_out, args.network_input, args, disc=True,
                                                          disc_image=not args.disc_kspace).to(args.device)

                    if first:
                        CONSTANT_PLOTS['measures'] = input.cpu()[2]
                        CONSTANT_PLOTS['mean'] = mean.cpu()[2]
                        CONSTANT_PLOTS['std'] = std.cpu()[2]
                        CONSTANT_PLOTS['gt'] = target_full[2]
                        first = False

                    # MAKE PREDICTIONS
                    real_pred = discriminator(disc_target_batch)
                    fake_pred = discriminator(disc_output_batch)

                    real_acc = real_pred[real_pred > 0].shape[0]
                    fake_acc = fake_pred[fake_pred <= 0].shape[0]

                    batch_loss['d_acc'].append((real_acc + fake_acc) / 32)

                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, disc_target_batch.data,
                                                                disc_output_batch.data, args)
                    # Adversarial loss
                    d_loss = torch.mean(fake_pred) - torch.mean(real_pred) + lambda_gp * gradient_penalty

                    d_loss.backward()
                    optimizer_D.step()

                optimizer_G.zero_grad()

                # Generate a batch of images
                z = torch.FloatTensor(
                    np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(0.001))).to(args.device)
                output_gen = generator(input_w_z.to(args.device), z)

                if args.network_input == 'kspace':
                    refined_out = output_gen + old_input[:]
                else:
                    refined_out = readd_measures_im(output_gen, old_input.to(args.device), args)

                disc_inp = prep_input_2_chan(refined_out, args.network_input, args, disc=True,
                                             disc_image=not args.disc_kspace)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(disc_inp)
                g_loss = -0.01 * torch.mean(fake_validity) - mssim_tensor(disc_target_batch, disc_inp)

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
                    input, target_full, mean, std, nnz_index_mask = data

                    input = prep_input_2_chan(input, args.network_input, args).to(args.device)
                    target_full = prep_input_2_chan(target_full, args.network_input, args).to(args.device)

                    z = torch.FloatTensor(
                        np.random.normal(size=(input.shape[0], args.latent_size), scale=np.sqrt(1))).to(args.device)

                    output_gen = generator(input, z)

                    refined_out = readd_measures_im(output_gen, input, args)

                    ims = prep_input_2_chan(refined_out, args.network_input, args, disc=True,
                                            disc_image=not args.disc_kspace).permute(0, 2, 3, 1)
                    target_im = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(args.device).permute(
                        0, 2, 3, 1)

                    for k in range(ims.shape[0]):
                        output = complex_abs(ims[k])
                        target = complex_abs(target_im[k])

                        output = output.cpu().numpy() * std[k].numpy() + mean[k].numpy()
                        target = target.cpu().numpy() * std[k].numpy() + mean[k].numpy()

                        losses['ssim'].append(ssim(target, output))
                        losses['psnr'].append(psnr(target, output))

                    if i == 20:
                        break

            psnr_loss = np.mean(losses['psnr'])
            best_model = psnr_loss > best_loss_val
            best_loss_val = psnr_loss if psnr_loss > best_loss_val else best_loss_val

            GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
            GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
            GLOBAL_LOSS_DICT['d_acc'].append(np.mean(batch_loss['d_acc']))

            save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch]:.4f}] [Average D Acc: {GLOBAL_LOSS_DICT['d_acc'][epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch]:.4f}]\n"
            print(save_str)
            loss_file.write(save_str)

            save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
            print(save_str_2)

            save_model(args, epoch, generator, optimizer_G, best_loss_val, best_model, 'generator')
            save_model(args, epoch, discriminator, optimizer_D, best_loss_val, best_model, 'discriminator')

            plot_epoch(args, generator, epoch)
            generator.train()


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.FloatTensor
    inverse_mask = get_inverse_mask()

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

    # try:
    main(args)
    save_metrics(args)
    # except:
    #     save_metrics(args)
