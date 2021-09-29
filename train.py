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

# Tunable weight for gradient penalty
lambda_gp = 10


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


def prep_discriminator_input(data_tensor, num_vals, unet_type, inds=None):
    disc_inp = torch.zeros(num_vals, 2, 384, 384)

    if unet_type == 'kspace':
        for k in range(num_vals):
            output = torch.squeeze(data_tensor[k]) if not inds else torch.squeeze(data_tensor[inds[k]])

            output_tensor = torch.zeros(8, 384, 384, 2)
            output_tensor[:, :, :, 0] = output[0:8, :, :]
            output_tensor[:, :, :, 1] = output[8:16, :, :]

            output_x = ifft2c_new(output_tensor)
            output_x = transforms.root_sum_of_squares(output_x)

            disc_inp[k, :, :, :] = output_x.permute(2, 0, 1)
    else:
        raise NotImplementedError

    return disc_inp


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best, m_type):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / f'{m_type}_model.pt', exp_dir / f'{m_type}_best_model.pt')


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
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
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    args.in_chans = 17 if args.z_location == 3 else 16
    args.out_chans = 16

    if args.resume:
        generator, discriminator, args, best_dev_loss, start_epoch = resume_train(args)
    else:
        generator, discriminator, best_dev_loss, start_epoch = fresh_start(args)

    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader = create_data_loaders(args)
    lr = 10e-4
    beta_1 = 0.5
    beta_2 = 0.999

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))

    for epoch in range(start_epoch, args.num_epochs):
        for i, data in enumerate(train_loader):
            input, target_full, mean, std, nnz_index_mask = data
            old_input = input.to(args.device)
            input_w_z = add_z_to_input(args, input)

            for j in range(args.num_iters_discriminator):
                i_true = np.random.randint(0, target_full.shape[0], args.batch_size // 2)
                i_fake = np.random.randint(0, target_full.shape[0], args.batch_size // 2)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                if args.network_input == 'image':
                    raise NotImplementedError

                input_w_z = input_w_z.to(args.device)
                output_gen = generator(input_w_z)
                print('Gen done')
                temp = prep_discriminator_input(target_full, args.batch_size, args.network_input).to(args.device)
                temp_dis = discriminator(temp)
                print(output_gen.shape)
                print(temp_dis)
                exit()

                # TURN OUTPUT INTO IMAGE FOR DISCRIMINATION AND GET REAL IMAGES FOR DISCRIMINATION
                if args.network_input == 'kspace':
                    refined_out = output_gen + old_input
                else:
                    raise NotImplementedError

                disc_target_batch = prep_discriminator_input(target_full, args.batch_size // 2, args.network_input, inds=i_true)
                disc_output_batch = prep_discriminator_input(refined_out, args.batch_size // 2, args.network_input, inds=i_fake)

                real_pred = discriminator(disc_target_batch)
                fake_pred = discriminator(disc_output_batch)

                # Gradient penalty - TODO: FIX THIS
                gradient_penalty = compute_gradient_penalty(discriminator, disc_target_batch.data, disc_output_batch.data)

                # Adversarial loss
                d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

            optimizer_G.zero_grad()

            # Generate a batch of images
            output_gen = generator(input_w_z)

            if args.network_input == 'kspace':
                refined_out = output_gen + old_input
            else:
                raise NotImplementedError

            disc_inp = prep_discriminator_input(refined_out, args.batch_size, args.network_input)

            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(disc_inp)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.num_epochs, d_loss.item(), g_loss.item())
            )


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
    main(args)
