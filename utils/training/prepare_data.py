import torch

import numpy as np

from espirit import ifft, fft
from torch.utils.data import DataLoader
from data import transforms

from data.mri_data import SelectiveSliceData, SelectiveSliceData_Val
from utils.fftc import ifft2c_new, fft2c_new
import cv2


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, use_seed=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.use_seed = use_seed
        self.args = args
        self.mask = None

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # GRO Sampling mask:
        a = np.array(
            [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
             151, 155, 158, 161, 164,
             167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
             223, 226, 229, 233, 236,
             240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
             374])
        # a = np.array(
        #     [1, 24, 45, 64, 81, 97, 111, 123, 134, 144, 153, 161, 168, 175, 181, 183, 184, 185, 186, 187, 188, 189, 190,
        #      191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 205, 211, 218, 225, 233, 242, 252, 263, 275, 289,
        #      305, 322, 341, 362]
        # )
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 176:208] = True
        samp = m
        # m = np.zeros((96, 96))
        # m[:, a] = True
        # m[:, 42:54] = True
        # samp = m
        numcoil = 16
        mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
        mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

        kspace = kspace.transpose(1, 2, 0)

        x = ifft(kspace, (0, 1))  # (768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

        im_tensor = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)
        # output_x = transforms.root_sum_of_squares(im_tensor)
        # # REMOVE BELOW TWO LINES TO GO BACK UP
        # output_x_r = cv2.resize(output_x[:, :, 0].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        # output_x_c = cv2.resize(output_x[:, :, 1].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        #
        # output_x_r = torch.from_numpy(output_x_r).type(torch.FloatTensor).unsqueeze(-1)
        # output_x_c = torch.from_numpy(output_x_c).type(torch.FloatTensor).unsqueeze(-1)
        # ######################################
        # output_x = torch.cat((output_x_r, output_x_c), dim=-1)

        true_image = torch.clone(im_tensor)
        true_measures = fft2c_new(im_tensor) * mask
        image = im_tensor

        if self.args.inpaint or self.args.dynamic_inpaint:
            from random import randrange

            n = image.shape[1]
            square_length = n // 5
            end = n - square_length

            if self.args.dynamic_inpaint:
                rand_start_col = randrange(0, end)
                rand_start_row = randrange(0, end)
            else:
                rand_start_col = 5 * n // 8
                rand_start_row = 5 * n // 8

            image[rand_start_row:rand_start_row + square_length, rand_start_col:rand_start_col + square_length, :] = 0

        kspace = fft2c_new(image)
        masked_kspace = kspace * mask
        zfr = ifft2c_new(masked_kspace)

        # masked_kspace = kspace * mask

        # kspace = transforms.to_tensor(gt_kspace)
        # kspace = kspace.permute(2, 0, 1, 3)

        # masked_kspace = transforms.to_tensor(masked_kspace)
        # masked_kspace = masked_kspace.permute(2, 0, 1, 3)

        ###################################

        stacked_masked_zfr = torch.zeros(numcoil * 2, 384, 384)

        stacked_masked_zfr[0:numcoil, :, :] = torch.squeeze(zfr[:, :, :, 0])
        stacked_masked_zfr[numcoil:numcoil * 2, :, :] = torch.squeeze(zfr[:, :, :, 1])
        stacked_masked_zfr, mean, std = transforms.normalize_instance(stacked_masked_zfr)
        # zfr, mean, std = transforms.normalize_instance(zfr)

        stacked_image = torch.zeros(numcoil * 2, 384, 384)
        stacked_image[0:numcoil, :, :] = torch.squeeze(true_image[:, :, :, 0])
        stacked_image[numcoil:numcoil * 2, :, :] = torch.squeeze(true_image[:, :, :, 1])
        stacked_image = transforms.normalize(stacked_image, mean, std)
        # target = transforms.normalize(true_image, mean, std)
        true_me = transforms.normalize(ifft2c_new(true_measures), mean, std)

        temp = torch.zeros(numcoil, 384, 384, 2)
        stacked_masked_kspace = torch.zeros(numcoil * 2, 384, 384)
        temp[:, :, :, 0] = stacked_masked_zfr[0:numcoil, :, :]
        temp[:, :, :, 1] = stacked_masked_zfr[numcoil:numcoil * 2, :, :]
        masked_kspace_normalized = fft2c_new(temp)
        stacked_masked_kspace[0:numcoil, :, :] = torch.squeeze(masked_kspace_normalized[:, :, :, 0])
        stacked_masked_kspace[numcoil:numcoil * 2, :, :] = torch.squeeze(masked_kspace_normalized[:, :, :, 1])

        temp = torch.zeros(numcoil, 384, 384, 2)
        stacked_kspace = torch.zeros(numcoil * 2, 384, 384)
        temp[:, :, :, 0] = stacked_image[0:numcoil, :, :]
        temp[:, :, :, 1] = stacked_image[numcoil:numcoil * 2, :, :]
        kspace_normalized = fft2c_new(temp)
        stacked_kspace[0:numcoil, :, :] = torch.squeeze(kspace_normalized[:, :, :, 0])
        stacked_kspace[numcoil:numcoil * 2, :, :] = torch.squeeze(kspace_normalized[:, :, :, 1])

        temp = torch.zeros(numcoil, 384, 384, 2)
        temp[:, :, :, 0] = stacked_masked_zfr[0:numcoil, :, :]
        temp[:, :, :, 1] = stacked_masked_zfr[numcoil:numcoil * 2, :, :]
        true_measures_normal = fft2c_new(temp)

        return stacked_masked_kspace.permute(1, 2, 0), stacked_kspace.permute(1, 2, 0), mean, std, true_measures_normal


def create_datasets(args, val_only):
    if not val_only:
        train_data = SelectiveSliceData(
            root=args.data_path / 'multicoil_train',
            transform=DataTransform(args),
            challenge='multicoil',
            sample_rate=1,
            use_top_slices=True,
            number_of_top_slices=args.num_of_top_slices,
            restrict_size=False,
        )

    dev_data = SelectiveSliceData_Val(
        root=args.data_path / 'multicoil_val',
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
    )

    return dev_data, train_data if not val_only else None


def create_data_loaders(args, val_only=False):
    dev_data, train_data = create_datasets(args, val_only)

    if not val_only:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
        )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader if not val_only else None, dev_loader


# Helper functions for Transform
def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t


def unflatten(t, shape_t):
    t = t.reshape(shape_t)
    return t


def ImageCropandKspaceCompression(x):
    w_from = (x.shape[0] - 384) // 2  # crop images into 384x384
    h_from = (x.shape[1] - 384) // 2
    w_to = w_from + 384
    h_to = h_from + 384
    cropped_x = x[w_from:w_to, h_from:h_to, :]
    if cropped_x.shape[-1] > 16:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:16].reshape(384, 384, 16)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x
