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
        # a = np.array(
        #     [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
        #      151, 155, 158, 161, 164,
        #      167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
        #      223, 226, 229, 233, 236,
        #      240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
        #      374])
        a = np.array(
            [1, 9, 15, 21, 26, 31, 35, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59, 63, 67, 72, 77,
             83, 89]
        )
        # m = np.zeros((384, 384))
        # m[:, a] = True
        # m[:, 176:208] = True
        # samp = m
        m = np.zeros((96, 96))
        m[:, a] = True
        m[:, 42:54] = True
        samp = m
        numcoil = 2
        mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).transpose((1, 2, 0)).astype(np.float32))

        kspace = kspace.transpose(1, 2, 0)

        x = ifft(kspace, (0, 1))  # (768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

        im_tensor = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)
        output_x = transforms.root_sum_of_squares(im_tensor)
        # REMOVE BELOW TWO LINES TO GO BACK UP
        output_x_r = cv2.resize(output_x[:, :, 0].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)
        output_x_c = cv2.resize(output_x[:, :, 1].numpy(), dsize=(96, 96), interpolation=cv2.INTER_LINEAR)

        output_x_r = torch.from_numpy(output_x_r).unsqueeze(-1)
        output_x_c = torch.from_numpy(output_x_c).unsqueeze(-1)
        ######################################
        output_x = torch.cat((output_x_r, output_x_c), dim=-1)

        image = output_x

        if self.args.inpaint:
            from random import randrange

            n = image.shape[0]
            square_length = n // 5
            end = n - square_length

            rand_start = randrange(square_length, end)

            image[rand_start:rand_start + square_length, rand_start:rand_start + square_length, :] = 0

        kspace = fft2c_new(image)
        masked_kspace = kspace * mask

        # masked_kspace = kspace * mask

        # kspace = transforms.to_tensor(gt_kspace)
        # kspace = kspace.permute(2, 0, 1, 3)

        # masked_kspace = transforms.to_tensor(masked_kspace)
        # masked_kspace = masked_kspace.permute(2, 0, 1, 3)

        ###################################

        # stacked_masked_kspace = torch.zeros(16, 384, 384)
        #
        # stacked_masked_kspace[0:8, :, :] = torch.squeeze(masked_kspace[:, :, :, 0])
        # stacked_masked_kspace[8:16, :, :] = torch.squeeze(masked_kspace[:, :, :, 1])
        # stacked_masked_kspace, mean, std = transforms.normalize_instance(stacked_masked_kspace, eps=1e-11)
        stacked_masked_kspace, mean, std = transforms.normalize_instance(masked_kspace, eps=1e-11)
        # stacked_masked_kspace = (stacked_masked_kspace - (-4.0156e-11)) / (2.5036e-05)

        # stacked_kspace = torch.zeros(16, 384, 384)
        # stacked_kspace[0:8, :, :] = torch.squeeze(kspace[:, :, :, 0])
        # stacked_kspace[8:16, :, :] = torch.squeeze(kspace[:, :, :, 1])
        # stacked_kspace = transforms.normalize(stacked_kspace, mean, std, eps=1e-11)
        stacked_kspace = transforms.normalize(kspace, mean, std, eps=1e-11)
        # stacked_kspace = (stacked_kspace - (-4.0156e-11)) / (2.5036e-05)

        # mean = (-4.0156e-11)
        # std = (2.5036e-05)

        return stacked_masked_kspace, stacked_kspace, mean, std, None


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
            num_workers=24,
            pin_memory=True,
        )

    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=24,
        pin_memory=True,
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
    if cropped_x.shape[-1] >= 8:
        x_tocompression = cropped_x.reshape(384 ** 2, cropped_x.shape[-1])
        U, S, Vh = np.linalg.svd(x_tocompression, full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:, 0:8].reshape(384, 384, 8)
    else:
        coil_compressed_x = cropped_x

    return coil_compressed_x
