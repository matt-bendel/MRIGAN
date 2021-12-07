import torch
import cv2

import numpy as np

from data import transforms
from utils.fftc import ifft2c_new, fft2c_new


def get_mask():
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
    samp = m
    mask = np.tile(samp, (1, 1, 1)).transpose((1, 2, 0)).astype(np.float32)
    mask = cv2.resize(mask[:, :, 0], dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
    mask = np.repeat(np.expand_dims(mask, 2), 2, axis=2).transpose(2, 0, 1)
    # mask = np.repeat(np.expand_dims(mask, 0).transpose(0, 3, 1, 2), b_size, axis=0)

    return transforms.to_tensor(mask)


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
    Tensor = torch.FloatTensor
    inverse_mask = get_inverse_mask()
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


def readd_measures_im(data_tensor, old, args, kspace=False):
    im_size = 96
    disc_inp = torch.zeros(data_tensor.shape[0], 2, im_size, im_size).to(args.device)

    for k in range(data_tensor.shape[0]):
        output = torch.squeeze(data_tensor[k])
        output_tensor = fft2c_new(output.permute(1, 2, 0))

        old_out = torch.squeeze(old[k])
        old_out = fft2c_new(old_out.permute(1, 2, 0))

        # mask = get_mask()

        # disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1) * mask.to(args.device) + old_out.permute(2, 0, 1)
        disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1) + old_out.permute(2, 0, 1)

    for k in range(data_tensor.shape[0]):
        output = torch.squeeze(disc_inp[k])
        output_tensor = ifft2c_new(output.permute(1, 2, 0))

        disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1)

    return disc_inp


def prep_input_2_chan(data_tensor, unet_type, args, disc=False, disc_image=True):
    im_size = 96
    disc_inp = torch.zeros(data_tensor.shape[0], 2, im_size, im_size).to(args.device)

    if disc and disc_image:
        for k in range(data_tensor.shape[0]):
            output = torch.squeeze(data_tensor[k])
            if args.network_input == 'kspace':
                output_tensor = ifft2c_new(output.permute(1, 2, 0))
                disc_inp[k, :, :, :] = output_tensor.permute(2, 0, 1)
            else:
                disc_inp[k, :, :, :] = output

        return disc_inp

    disc_inp = ifft2c_new(data_tensor)
    return disc_inp.permute(0, 3, 1, 2)
