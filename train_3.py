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

from typing import Optional
from data import transforms
from utils.math import complex_abs
from utils.training.prepare_data import create_data_loaders
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
    # ssim_loss = pytorch_ssim.SSIM()
    return pytorch_msssim.msssim(gt, pred)


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


def save_metrics(args):
    with open(f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/saved_metrics/loss_{args.network_input}_{args.z_location}.pkl',
              'wb') as f:
        pickle.dump(GLOBAL_LOSS_DICT, f, pickle.HIGHEST_PROTOCOL)


def average(gen_tensor):
    average_tensor = torch.zeros((gen_tensor.shape[0], gen_tensor.shape[2], gen_tensor.shape[3], gen_tensor.shape[4])).to(gen_tensor.device)
    for j in range(gen_tensor.shape[0]):
        for i in range(gen_tensor.shape[1]):
            average_tensor[j, :, :, :] = torch.add(gen_tensor[j, i, :, :, :], average_tensor[j, :, :, :])

    return torch.div(average_tensor, gen_tensor.shape[1])


def average_gen(generator, input_w_z, old_input, args, true_measures):
    average_gen = torch.zeros(input_w_z.shape).to(args.device)
    gen_list = []
    for j in range(8):
        z = torch.FloatTensor(np.random.normal(size=(input_w_z.shape[0], args.latent_size), scale=np.sqrt(1))).to(
            args.device)
        output_gen = generator(input=input_w_z, z=z)

        if args.network_input == 'kspace':
            # refined_out = output_gen + old_input[:, 0:16]
            refined_out = output_gen + old_input[:]
        else:
            refined_out = readd_measures_im(output_gen, old_input, args, true_measures=true_measures) if not args.inpaint else output_gen

        gen_list.append(refined_out)
        average_gen = torch.add(average_gen, refined_out)

    return torch.div(average_gen, 8), gen_list


def get_gen_supervised(args):
    from utils.training.prepare_model import build_model, build_optim, build_discriminator

    checkpoint_file_gen = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/image/{args.z_location}/generator_model.pt')
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    checkpoint_file_dis = pathlib.Path(
        f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/image/{args.z_location}/discriminator_model.pt')
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

    return generator, opt_gen, discriminator, opt_dis, args, checkpoint_gen['best_dev_loss'], checkpoint_dis['best_dev_loss'], checkpoint_gen['epoch']


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 16
    args.out_chans = 16

    if args.resume:
        generator, optimizer_G, discriminator, optimizer_D, args, best_loss_val, best_loss_dis, start_epoch = get_gen_supervised(args)
        start_epoch = start_epoch + 1
    else:
        generator, discriminator, best_dev_loss, start_epoch = fresh_start(args)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        best_loss_val = 0
        best_loss_dis = 0

    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader = create_data_loaders(args)

    with open(f'trained_models/{args.network_input}/loss_{args.z_location}.txt', 'w') as loss_file:
        for epoch in range(start_epoch, args.num_epochs):
            batch_loss = {
                'g_loss': [],
                'd_loss': [],
                'd_acc': []
            }

            for i, data in enumerate(train_loader):
                input, target_full, mean, std, true_measures = data

                input = prep_input_2_chan(input, args.network_input, args)
                target_full = prep_input_2_chan(target_full, args.network_input, args).to(args.device)
                true_measures = true_measures.to(args.device)

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
                        refined_out = readd_measures_im(output_gen, old_input, args, true_measures=true_measures) if not args.inpaint else output_gen

                    # TURN OUTPUT INTO IMAGE FOR DISCRIMINATION AND GET REAL IMAGES FOR DISCRIMINATION
                    disc_target_batch = prep_input_2_chan(target_full, args.network_input, args, disc=True,
                                                          disc_image=not args.disc_kspace).to(
                        args.device)
                    disc_output_batch = prep_input_2_chan(refined_out, args.network_input, args, disc=True,
                                                          disc_image=not args.disc_kspace).to(args.device)

                    # MAKE PREDICTIONS
                    real_pred = discriminator(input=disc_target_batch, y=old_input)
                    fake_pred = discriminator(input=disc_output_batch, y=old_input)

                    real_acc = real_pred[real_pred > 0].shape[0]
                    fake_acc = fake_pred[fake_pred <= 0].shape[0]

                    batch_loss['d_acc'].append((real_acc + fake_acc) / real_pred.shape[0])

                    # Gradient penalty
                    gradient_penalty = compute_gradient_penalty(discriminator, disc_target_batch.data,
                                                                disc_output_batch.data, args, old_input.data)
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
                args.num_z, old_input.shape[0], old_input.shape[1], old_input.shape[2], old_input.shape[3])).to(
                    args.device)
                for k in range(args.num_z):
                    output_gen[k, :, :, :, :] = generator(input_w_z, z[k])

                if args.network_input == 'kspace':
                    refined_out = output_gen + old_input[:]
                else:
                    refined_out = torch.zeros(size=output_gen.shape).to(args.device)
                    for k in range(args.num_z):
                        refined_out[k, :, :, :, :] = readd_measures_im(output_gen[k], old_input, args, true_measures=true_measures) if not args.inpaint else output_gen[k]

                disc_output_batch = torch.zeros(size=refined_out.shape).to(args.device)
                for k in range(args.num_z):
                    disc_output_batch[k, :, :, :, :] = prep_input_2_chan(refined_out[k], args.network_input, args, disc=True,
                                                             disc_image=not args.disc_kspace).to(args.device)

                disc_inputs_gen = torch.zeros(
                    size=(old_input.shape[0], args.num_z, disc_output_batch.shape[2], disc_output_batch.shape[3],
                          disc_output_batch.shape[4])
                ).to(args.device)
                for l in range(old_input.shape[0]):
                    for k in range(args.num_z):
                        disc_inputs_gen[l, k, :, :, :] = disc_output_batch[k, l, :, :, :]

                avg_recon = torch.mean(disc_inputs_gen, dim=1) #average(disc_inputs_gen)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_pred = torch.zeros((old_input.shape[0], args.num_z)).to(args.device)
                for k in range(old_input.shape[0]):
                    cond = torch.zeros(1, disc_inputs_gen.shape[2], disc_inputs_gen.shape[3], disc_inputs_gen.shape[4])
                    cond[0, :, :, :] = old_input[k, :, :, :]
                    cond = cond.repeat(args.num_z, 1, 1, 1)
                    temp = discriminator(input=disc_inputs_gen[k], y=cond)
                    fake_pred[k] = temp[:, 0]

                gen_pred_loss = torch.mean(fake_pred[0])
                for k in range(old_input.shape[0] - 1):
                    gen_pred_loss += torch.mean(fake_pred[k + 1])

                # var_loss = torch.mean(torch.var(disc_inputs_gen, dim=1) * torch.abs(target_full-avg_recon), dim=(0, 1, 2, 3))
                var_loss = torch.mean(torch.var(disc_inputs_gen, dim=1), dim=(0, 1, 2, 3))

                var_weight = 0.075
                adv_weight = 1e-6
                mssim_weight = 0.84

                # TODO: BEST -0.001 adv and var_weight = 0.012
                g_loss = -adv_weight*torch.mean(gen_pred_loss) + (1-mssim_weight) * F.l1_loss(target_full, avg_recon) - mssim_weight * mssim_tensor(
                    target_full, avg_recon) - var_weight * var_loss

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
                    input, target_full, mean, std, true_measures = data

                    input = prep_input_2_chan(input, args.network_input, args).to(args.device)
                    target_full = prep_input_2_chan(target_full, args.network_input, args).to(args.device)
                    true_measures = true_measures.to(args.device)

                    output_gen, gen_list = average_gen(generator, input, input, args, true_measures)

                    ims = prep_input_2_chan(output_gen, args.network_input, args, disc=True,
                                            disc_image=not args.disc_kspace)
                    target_im = prep_input_2_chan(target_full, args.network_input, args, disc=True).to(
                        args.device)

                    for k in range(ims.shape[0]):
                        output_rss = torch.zeros(8, ims.shape[2], ims.shape[2], 2)
                        output_rss[:, :, :, 0] = ims[k, 0:8, :, :]
                        output_rss[:, :, :, 1] = ims[k, 8:16, :, :]
                        output = transforms.root_sum_of_squares(complex_abs(output_rss * std[k] + mean[k]))

                        target_rss = torch.zeros(8, target_im.shape[2], target_im.shape[2], 2)
                        target_rss[:, :, :, 0] = target_im[k, 0:8, :, :]
                        target_rss[:, :, :, 1] = target_im[k, 8:16, :, :]
                        target = transforms.root_sum_of_squares(complex_abs(target_rss * std[k] + mean[k]))

                        output = output.cpu().numpy()
                        target = target.cpu().numpy()

                        losses['ssim'].append(ssim(target, output))
                        losses['psnr'].append(psnr(target, output))

                        if i + 1 == 1 and k == 2:
                            gen_im_list = []
                            for r, val in enumerate(gen_list):
                                val_rss = torch.zeros(8, output.shape[0], output.shape[0], 2).to(args.device)
                                val_rss[:, :, :, 0] = val[k, 0:8, :, :]
                                val_rss[:, :, :, 1] = val[k, 8:16, :, :]
                                gen_im_list.append(transforms.root_sum_of_squares(complex_abs(val_rss * std[k] + mean[k])).cpu().numpy())

                            std_dev = np.zeros(output.shape)
                            for val in gen_im_list:
                                std_dev = std_dev + np.power((val - output), 2)

                            std_dev = std_dev / args.num_z
                            std_dev = np.sqrt(std_dev)

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

                            plt.savefig('std_dev_gen.png')
                            plt.close()

                            plt.figure()
                            plt.imshow(np.abs(output), cmap='gray')
                            plt.savefig('temp_gen_out.png')
                            plt.close()

                            plt.figure()
                            plt.imshow(np.abs(target), cmap='gray')
                            plt.savefig('temp_gen_targ.png')
                            plt.close()

                    if i == 75:
                        break

            psnr_loss = np.mean(losses['psnr'])
            best_model = psnr_loss > best_loss_val
            best_loss_val = psnr_loss if psnr_loss > best_loss_val else best_loss_val

            GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
            GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
            GLOBAL_LOSS_DICT['d_acc'].append(np.mean(batch_loss['d_acc']))

            save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch-start_epoch]:.4f}] [Average D Acc: {GLOBAL_LOSS_DICT['d_acc'][epoch-start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch-start_epoch]:.4f}]\n"
            print(save_str)
            loss_file.write(save_str)

            save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
            print(save_str_2)

            save_model(args, epoch, generator, optimizer_G, best_loss_val, best_model, 'generator')
            save_model(args, epoch, discriminator, optimizer_D, best_loss_val, best_model, 'discriminator')

            # plot_epoch(args, generator, epoch)
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

    # try:
    main(args)
    save_metrics(args)
    # except:
    #     save_metrics(args)
