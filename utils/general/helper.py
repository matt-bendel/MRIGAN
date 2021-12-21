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
    m = np.zeros((384, 384))
    m[:, a] = True
    m[:, 176:208] = True
    # a = np.array(
    #     [1, 9, 15, 21, 26, 31, 35, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59, 63, 67, 72, 77,
    #      83, 89]
    # )
    # m = np.zeros((96, 96))
    # m[:, a] = True
    # m[:, 42:54] = True

    # mask = np.repeat(np.expand_dims(mask, 0).transpose(0, 3, 1, 2), b_size, axis=0)

    return np.where(m == 1)


def add_z_to_input(args, input):
    """
                    0: No latent vector
                    1: Add latent vector to zero filled areas
                    2: Add latent vector to middle of network (between encoder and decoder)
                    3: Add as an extra input channel
                    """
    Tensor = torch.FloatTensor
    inverse_mask = get_mask()
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


def readd_measures_im(data_tensor, old, args, kspace=False, true_measures=False):
    im_size = 384
    disc_inp = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)
    kspace_no_ip = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)

    for k in range(data_tensor.shape[0]):
        temp = torch.zeros(8, im_size, im_size, 2).to(args.device)
        temp[:, :, :, 0] = data_tensor[k, 0:8, :, :]
        temp[:, :, :, 1] =  data_tensor[k, 8:16, :, :]
        output_tensor = fft2c_new(temp)

        temp = torch.zeros(8, im_size, im_size, 2).to(args.device)
        temp[:, :, :, 0] = old[k, 0:8, :, :]
        temp[:, :, :, 1] = old[k, 8:16, :, :]
        old_out = fft2c_new(temp)

        inds = get_mask()

        if not args.dynamic_inpaint and not args.inpaint:
            refined_measures = output_tensor
            refined_measures[:, ind[0], ind[1], :] = old_out[:, inds[0], inds[1], :]
        elif not args.inpaint:
            refined_measures = output_tensor
            refined_measures[:, ind[0], ind[1], :] = true_measures[k, :, inds[0], inds[1], :]
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
    im_size = 384
    disc_inp = torch.zeros(data_tensor.shape[0], 16, im_size, im_size).to(args.device)

    if disc and disc_image:
        return data_tensor

    temp = torch.zeros(data_tensor.shape[0], 8, im_size, im_size, 2).to(args.device)
    new_data = data_tensor.permute(0,3,1,2)
    temp[:,:,:,:,0] = new_data[:, 0:8, :, :]
    temp[:,:,:,:,1] = new_data[:, 8:16, :, :]
    temp = ifft2c_new(temp)
    disc_inp[:, 0:8, :, :] = temp[:,:,:,:,0]
    disc_inp[:, 8:16, :, :] = temp[:,:,:,:,1]
    return disc_inp
