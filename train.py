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
import random
import os
import shutil
import time
import torch

import numpy as np
import torch.autograd as autograd

from data import transforms
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.evaluate import nmse
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

# Tunable weight for gradient penalty
lambda_gp = 10

GLOBAL_LOSS_DICT = {
    'g_loss': [],
    'd_loss': [],
    'mSSIM': []
}
CONSTANT_PLOTS = {
    'measures': None,
    'measures_w_z': None,
    'mean': None,
    'std': None,
    'gt': None
}

def get_inverse_mask():
    a = np.array(
        [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
         151, 155, 158, 161, 164,
         167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
         223, 226, 229, 233, 236,
         240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
         374])
    m = np.ones((384, 384))
    m[:, a] = 0
    m[:, 176:208] = 0

    return m


def add_z_to_input(args, input):
    """
                    0: No latent vector
                    1: Add latent vector to zero filled areas
                    2: Add latent vector to middle of network (between encoder and decoder)
                    3: Add as an extra input channel
                    """
    for i in range(input.shape[0]):
        if args.z_location == 1 or args.z_location == 3:
            z = np.random.normal(size=(384, 384))
            z = Tensor(z * inverse_mask) if args.z_location == 1 else Tensor(z)
            if args.z_location == 1:
                for val in range(input.shape[1]):
                    input[i, val, :, :] = input[i, val, :, :].add(z)
            else:
                input[i, 16, :, :] = z

    return input


def prep_discriminator_input(data_tensor, num_vals, unet_type, indvals, inds=None, mean=None, std=None):
    disc_inp = torch.zeros(data_tensor.shape[0], 2, 384, 384)

    if unet_type == 'kspace':
        for k in range(data_tensor.shape[0]):
            # output = torch.squeeze(data_tensor[k]) if not inds else torch.squeeze(data_tensor[indvals[k]])
            # data_tensor = data_tensor * std[k] + mean[k] if not inds else data_tensor * std[indvals[k]] + mean[indvals[k]]

            output = torch.squeeze(data_tensor[k]) if not inds else torch.squeeze(data_tensor[indvals[k]])
            data_tensor = data_tensor if not inds else data_tensor
            output_tensor = torch.zeros(8, 384, 384, 2)
            output_tensor[:, :, :, 0] = output[0:8, :, :]
            output_tensor[:, :, :, 1] = output[8:16, :, :]

            output_x = ifft2c_new(output_tensor)
            output_x = transforms.root_sum_of_squares(output_x)

            disc_inp[k, :, :, :] = output_x.permute(2, 0, 1)
    else:
        raise NotImplementedError

    return disc_inp


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


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 17 if args.z_location == 3 else 16
    args.out_chans = 16

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

    loss_file = open(f'trained_models/{args.network_input}/loss_{args.z_location}.txt', 'w')

    for epoch in range(start_epoch, args.num_epochs):
        batch_loss = {
            'g_loss': [],
            'd_loss': []
        }
        for i, data in enumerate(train_loader):
            input, target_full, mean, std, nnz_index_mask = data
            old_input = input.to(args.device)

            for j in range(args.num_iters_discriminator):
                i_true = np.random.randint(0, target_full.shape[0], args.batch_size // 2)
                i_fake = np.random.randint(0, target_full.shape[0], args.batch_size // 2)
                input_w_z = add_z_to_input(args, input)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # TODO: TRANSFORM INTO 16 CHANNEL IMAGE
                if args.network_input == 'image':
                    raise NotImplementedError

                input_w_z = input_w_z.to(args.device)
                if args.z_location == 2:
                    output_gen = generator(input_w_z, device=args.device, latent_size=args.latent_size)
                else:
                    output_gen = generator(input_w_z)

                if args.network_input == 'kspace':
                    refined_out = output_gen + old_input[:, 0:16]
                else:
                    # TODO: TRANSFORM IMAGE BACK TO K-SPACE AND ADD OLD OUT
                    raise NotImplementedError

                # TURN OUTPUT INTO IMAGE FOR DISCRIMINATION AND GET REAL IMAGES FOR DISCRIMINATION
                disc_target_batch = prep_discriminator_input(target_full.to(args.device), args.batch_size,
                                                             args.network_input,
                                                             i_true, inds=False, mean=mean, std=std).to(args.device)
                disc_output_batch = prep_discriminator_input(refined_out, args.batch_size, args.network_input,
                                                             i_fake, inds=False, mean=mean, std=std).to(args.device)

                # PLOT VERY FIRST GENERATED IMAGE
                if first:
                    CONSTANT_PLOTS['measures'] = input.cpu()[2]
                    CONSTANT_PLOTS['measures_w_z'] = input_w_z.detach().cpu()[2]
                    CONSTANT_PLOTS['mean'] = mean.cpu()[2]
                    CONSTANT_PLOTS['std'] = std.cpu()[2]
                    CONSTANT_PLOTS['gt'] = target_full.cpu()[2]

                    im_check = complex_abs(disc_output_batch[2].permute(1, 2, 0))
                    im_np = im_check.detach().cpu().numpy()
                    plt.imshow(np.abs(im_np), origin='lower', cmap='gray')
                    plt.savefig(f'first_gen_{args.network_input}_{args.z_location}.png')
                    first = False

                # MAKE PREDICTIONS
                real_pred = discriminator(disc_target_batch)
                fake_pred = discriminator(disc_output_batch)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, disc_target_batch.data,
                                                            disc_output_batch.data, args)
                # Adversarial loss
                d_loss = torch.mean(fake_pred) - torch.mean(real_pred) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()

            # Generate a batch of images
            output_gen = generator(input_w_z.to(args.device))

            if args.network_input == 'kspace':
                refined_out = output_gen + old_input[:, 0:16]
            else:
                raise NotImplementedError

            disc_inp = prep_discriminator_input(refined_out, args.batch_size, args.network_input, [], mean=mean,
                                                std=std)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(disc_inp)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Val mSSIM: %.4f]"
                % (epoch, args.num_epochs, i, len(train_loader.dataset)/args.batch_size, d_loss.item(), g_loss.item(), 0.0)
            )

        # TODO: ADD VALIDATION HERE - ONLY A SMALL SUBSET OF VAL DATA
        # for i, data in enumerate(train_loader):
        #     input, target_full, mean, std, nnz_index_mask = data
        #     old_input = input.to(args.device)

        best_model = True  # val_data()
        best_loss_val = 1e9  # val_data()
        ssim_loss = 0

        # TODO: ADD END OF EPOCH LOSS FROM VAL AND AVERAGE EPOCH LOSS
        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
        GLOBAL_LOSS_DICT['mSSIM'].append(ssim_loss)

        save_str = f"END OF EPOCH {epoch}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch]}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch]}] [Val mSSIM: {GLOBAL_LOSS_DICT['mSSIM'][epoch]}]\n"
        print(save_str)
        loss_file.write(save_str)

        save_model(args, epoch, generator, optimizer_G, best_loss_val, best_model, 'generator')
        save_model(args, epoch, discriminator, optimizer_D, best_loss_val, best_model, 'discriminator')

        if epoch + 1 % 5 == 0:
            std = CONSTANT_PLOTS['std']
            mean = CONSTANT_PLOTS['mean']

            plot_out = generator(CONSTANT_PLOTS['measures_w_z'].unsqueeze(0).to(args.device))

            if args.network_input == 'kspace':
                refined_out = plot_out.cpu() + CONSTANT_PLOTS['measures'][0:16].unsqueeze(0)
            else:
                raise NotImplementedError

            target_plot = prep_discriminator_input(CONSTANT_PLOTS['gt'].unsqueeze(0), 1, args.network_input, [], inds=False, mean=mean, std=std)[0]
            output_plot = prep_discriminator_input(refined_out, args.batch_size, args.network_input,
                                                         [], inds=False, mean=mean, std=std).to(args.device)[0]

            im_real = complex_abs(target_plot.permute(1, 2, 0)) * std + mean
            im_real_np = im_real.numpy()

            im_fake = complex_abs(output_plot.permute(1, 2, 0)) * std + mean
            im_fake_np = im_fake.detach().cpu().numpy()

            fig = plt.figure(figsize=(6,6))
            fig.suptitle(f'Generated and GT Images at Epoch {epoch}')
            ax = fig.add_subplot(2,2,1)
            ax.imshow(np.abs(im_real_np), origin='lower', cmap='gray', vmin=0, vmax=np.max(im_real_np))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel(f'GT')

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(np.abs(im_fake_np), origin='lower', cmap='gray', vmin=0, vmax=np.max(im_real_np))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel(f'Reconstruction')

            ax = fig.add_subplot(2, 2, 4)
            # MAY NEED TO REVISE BOUNDS BELOW
            k = 3
            ax.imshow(k*np.abs(im_real_np - im_fake_np), origin='lower', cmap='jet')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel(f'Relative Error')

            plt.savefig(f'/home/bendel.8/Git_Repos/MRIGAN/training_images/test_gen_{args.network_input}_{args.z_location}_{epoch+1}.png')

            if epoch + 1 == 10:
                exit()

    loss_file.close()


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

    # TODO: Add metric plotting from global dict
    # try:
    main(args)
        # SAVE METRICS
    # except:
    #     print("SAVE METRICS")
        #print(len(GLOBAL_LOSS_DICT['g_loss']))
