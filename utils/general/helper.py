import torch
import cv2

import numpy as np

from data import transforms
from utils.fftc import ifft2c_new, fft2c_new


def get_mask(im_size):
    if im_size == 384:
        a = np.array(
            [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
             151, 155, 158, 161, 164,
             167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
             223, 226, 229, 233, 236,
             240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
             374])
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 176:208] = True
    else:
        a = np.array(
            [1, 10, 18, 25, 31, 37, 42, 46, 50, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
             76, 80, 84, 88, 93, 99, 105, 112, 120])

        m = np.zeros((128, 128))
        m[:, a] = True
        m[:, 56:73] = True

    return np.where(m == 1)


def readd_measures_im(data_tensor, old, args, kspace=False, true_measures=False):
    im_size = args.im_size
    disc_inp = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)
    kspace_no_ip = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)

    for k in range(data_tensor.shape[0]):
        temp = torch.zeros(8, im_size, im_size, 2).to(args.device)
        temp[:, :, :, 0] = data_tensor[k, 0:8, :, :]
        temp[:, :, :, 1] = data_tensor[k, 8:16, :, :]
        output_tensor = fft2c_new(temp)

        temp = torch.zeros(8, im_size, im_size, 2).to(args.device)
        temp[:, :, :, 0] = old[k, 0:8, :, :]
        temp[:, :, :, 1] = old[k, 8:16, :, :]
        old_out = fft2c_new(temp)

        inds = get_mask(im_size)

        if not args.dynamic_inpaint and not args.inpaint:
            refined_measures = output_tensor
            refined_measures[:, inds[0], inds[1], :] = old_out[:, inds[0], inds[1], :]
        elif not args.inpaint:
            refined_measures = output_tensor
            refined_measures[:, inds[0], inds[1], :] = true_measures[k, :, inds[0], inds[1], :]
        else:
            refined_measures = output_tensor

        disc_inp[k, 0:8, :, :] = refined_measures[:, :, :, 0]
        disc_inp[k, 8:16, :, :] = refined_measures[:, :, :, 1]
        kspace_no_ip[k, 0:8, :, :] = output_tensor[:, :, :, 0]
        kspace_no_ip[k, 8:16, :, :] = output_tensor[:, :, :, 1]

    if kspace:
        return disc_inp if not args.inpaint else kspace_no_ip

    for k in range(data_tensor.shape[0]):
        temp = torch.zeros(8, im_size, im_size, 2).to(args.device)
        temp[:, :, :, 0] = disc_inp[k, 0:8, :, :]
        temp[:, :, :, 1] = disc_inp[k, 8:16, :, :]
        output_tensor = ifft2c_new(temp)

        disc_inp[k, 0:8, :, :] = output_tensor[:, :, :, 0]
        disc_inp[k, 8:16, :, :] = output_tensor[:, :, :, 1]

    return disc_inp


def prep_input_2_chan(data_tensor, unet_type, args, disc=False, disc_image=True):
    im_size = args.im_size
    disc_inp = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)

    if disc and disc_image:
        return data_tensor

    temp = torch.zeros(data_tensor.shape[0], 8, im_size, im_size, 2).to(args.device)
    new_data = data_tensor.permute(0, 3, 1, 2)
    temp[:, :, :, :, 0] = new_data[:, 0:8, :, :]
    temp[:, :, :, :, 1] = new_data[:, 8:16, :, :]
    temp = ifft2c_new(temp)
    disc_inp[:, 0:8, :, :] = temp[:, :, :, :, 0]
    disc_inp[:, 8:16, :, :] = temp[:, :, :, :, 1]
    return disc_inp