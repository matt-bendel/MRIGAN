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

from data import transforms
from utils.math import complex_abs
from utils.evaluate import nmse
from utils.training.prepare_data import create_data_loaders
from utils.training.parse_args import create_arg_parser
from utils.training.prepare_model import resume_train, fresh_start
from tensorboardX import SummaryWriter

from torch.nn import functional as F


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


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


# def train_epoch(args, epoch, model, data_loader, optimizer, writer):
#     model.train()
#     avg_loss = 0.
#     start_epoch = start_iter = time.perf_counter()
#     global_step = epoch * len(data_loader)
#     for iter, data in enumerate(data_loader):
#         input, target, mean, std, nnz_index_mask = data
#         input = input.to(args.device)
#
#         output = model(input)  # .squeeze(1)
#         output[:, :, :, nnz_index_mask] = 0
#
#         target[:, :, :, nnz_index_mask] = 0
#
#         target = target.to(args.device)
#
#         loss = F.l1_loss(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
#
#         writer.add_scalar('TrainLoss', loss.item(), global_step + iter)
#
#         if iter % args.report_interval == 0:
#             logging.info(
#                 f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
#                 f'Iter = [{iter:4d}/{len(data_loader):4d}] '
#                 f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
#                 f'Time = {time.perf_counter() - start_iter:.4f}s',
#             )
#
#         start_iter = time.perf_counter()
#
#     return avg_loss, time.perf_counter() - start_epoch
#
#
# def evaluate(args, epoch, model, data_loader, writer):
#     model.eval()
#     losses = []
#     start = time.perf_counter()
#
#     with torch.no_grad():
#         for iter, data in enumerate(data_loader):
#
#             input, target_full, mean, std = data
#             output_full = model(input.to(args.device))
#
#             for i in range(output_full.size(0)):
#                 target = torch.squeeze(target_full[i, :, :, :]).to(args.device)
#                 output = torch.squeeze(output_full[i, :, :, :])
#
#                 target = target * std[i] + mean[i]
#                 output = output * std[i] + mean[i]
#
#                 target_tensor = torch.zeros(8, 384, 384, 2)
#                 target_tensor[:, :, :, 0] = target[0:8, :, :]
#                 target_tensor[:, :, :, 1] = target[8:16, :, :]
#
#                 output_tensor = torch.zeros(8, 384, 384, 2)
#                 output_tensor[:, :, :, 0] = output[0:8, :, :]
#                 output_tensor[:, :, :, 1] = output[8:16, :, :]
#
#                 target_ifft_1 = complex_abs(target_tensor)
#                 target_ifft_2 = transforms.root_sum_of_squares(target_ifft_1)
#                 target_x = target_ifft_2.cpu().numpy()
#
#                 output_ifft_1 = complex_abs(output_tensor)
#                 output_ifft_2 = transforms.root_sum_of_squares(output_ifft_1)
#                 output_x = output_ifft_2.cpu().numpy()
#
#                 NMSE = nmse(target_x, output_x)
#                 losses.append(NMSE)
#
#         writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
#
#     return np.mean(losses), time.perf_counter() - start


def main(args):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    # TODO: ADD ABILITY TO RESUME
    # if args.resume:
    #     model, optimizer, args, best_dev_loss, start_epoch = resume_train(args)
    # else:

    generator, discriminator, best_dev_loss, start_epoch = fresh_start(args)

    logging.info(args)
    logging.info(generator)
    logging.info(discriminator)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    lr = 10e-4
    beta_1 = 0.5
    beta_2 = 0.999

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))

    inverse_mask = get_inverse_mask()
    for epoch in range(start_epoch, args.num_epochs):
        for i, data in enumerate(train_loader):
            input, target, mean, std, nnz_index_mask = data
            print(input.shape)
            exit()
            for j in range(args.num_iters_discriminator):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()
                """
                0: No latent vector
                1: Add latent vector to zero filled areas
                2: Add latent vector to middle of network (between encoder and decoder)
                3: Add as an extra input channel
                """
                if args.z_location == 1 or args.z_location == 3:
                    z = np.random.normal(size=(384,384))
                    z = Tensor(z * inverse_mask) if args.z_location == 1 else Tensor(z)

                    if args.z_location == 1:
                        for val in range(input.shape[0]):
                            input[val,:,:] = input[val, :, :].add(z)
                    else:
                        input[16, :, :] = z
                    # DEBUG
                    print(input)
                    exit()
                elif args.z_location == 2:
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                input = input.to(args.device)
                output = generator(input)

                real_pred = discriminator(data)
                fake_pred = discriminator(output)




    #     train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
    #     dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
    #
    #     is_new_best = dev_loss < best_dev_loss
    #     best_dev_loss = min(best_dev_loss, dev_loss)
    #     save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
    #     logging.info(
    #         f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
    #         f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
    #     )
    # writer.close()


if __name__ == '__main__':
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
