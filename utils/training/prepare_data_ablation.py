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
            [1, 10, 18, 25, 31, 37, 42, 46, 50, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
             76, 80, 84, 88, 93, 99, 105, 112, 120])

        m = np.zeros((128, 128))
        m[:, a] = True
        m[:, 56:73] = True
        samp = m
        numcoil = 8
        mask = transforms.to_tensor(np.tile(samp, (numcoil, 1, 1)).astype(np.float32))
        mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

        kspace = kspace.transpose(1, 2, 0)

        x = ifft(kspace, (0, 1))  # (768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

        complex_coil_im = transforms.to_tensor(coil_compressed_x).permute(2, 0, 1, 3)  # (8, 384, 384, 2)

        im_tensor = reduce_resolution(complex_coil_im)  # (8, 128, 128, 2)

        true_image = torch.clone(im_tensor)
        true_measures_im = ifft2c_new(fft2c_new(im_tensor) * mask)
        image = im_tensor

        if self.args.dynamic_inpaint:
            from random import randrange

            n = image.shape[1]
            square_length = n // 5
            end = n - square_length

            rand_start_col = randrange(0, end)
            rand_start_row = randrange(0, end)

            image[:, rand_start_row:rand_start_row + square_length, rand_start_col:rand_start_col + square_length,
            :] = 0

        if self.args.half_brain:
            poo = None

        kspace = fft2c_new(image)
        masked_kspace = kspace * mask
        zfr = ifft2c_new(masked_kspace)

        ###################################

        stacked_masked_zfr = torch.zeros(16, 128, 128)
        stacked_masked_zfr[0:8, :, :] = torch.squeeze(zfr[:, :, :, 0])
        stacked_masked_zfr[8:16, :, :] = torch.squeeze(zfr[:, :, :, 1])
        stacked_masked_zfr, mean, std = transforms.normalize_instance(stacked_masked_zfr)

        stacked_image = torch.zeros(16, 128, 128)
        stacked_image[0:8, :, :] = torch.squeeze(true_image[:, :, :, 0])
        stacked_image[8:16, :, :] = torch.squeeze(true_image[:, :, :, 1])
        stacked_image = transforms.normalize(stacked_image, mean, std)

        stacked_true_measures = torch.zeros(16, 128, 128)
        stacked_true_measures[0:8, :, :] = torch.squeeze(true_measures_im[:, :, :, 0])
        stacked_true_measures[8:16, :, :] = torch.squeeze(true_measures_im[:, :, :, 1])
        stacked_true_measures = transforms.normalize(stacked_true_measures, mean, std)

        temp = torch.zeros(8, 128, 128, 2)
        stacked_masked_kspace = torch.zeros(16, 128, 128)
        temp[:, :, :, 0] = stacked_masked_zfr[0:8, :, :]
        temp[:, :, :, 1] = stacked_masked_zfr[8:16, :, :]
        masked_kspace_normalized = fft2c_new(temp)
        stacked_masked_kspace[0:8, :, :] = torch.squeeze(masked_kspace_normalized[:, :, :, 0])
        stacked_masked_kspace[8:16, :, :] = torch.squeeze(masked_kspace_normalized[:, :, :, 1])

        temp = torch.zeros(8, 128, 128, 2)
        stacked_kspace = torch.zeros(16, 128, 128)
        temp[:, :, :, 0] = stacked_image[0:8, :, :]
        temp[:, :, :, 1] = stacked_image[8:16, :, :]
        kspace_normalized = fft2c_new(temp)
        stacked_kspace[0:8, :, :] = torch.squeeze(kspace_normalized[:, :, :, 0])
        stacked_kspace[8:16, :, :] = torch.squeeze(kspace_normalized[:, :, :, 1])

        temp = torch.zeros(8, 128, 128, 2)
        temp[:, :, :, 0] = stacked_true_measures[0:8, :, :]
        temp[:, :, :, 1] = stacked_true_measures[8:16, :, :]
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


def reduce_resolution(im):
    reduced_im = np.zeros((8, 128, 128, 2))
    for i in range(im.shape[0] // 2):
        reduced_im[i, :, :, 0] = cv2.resize(im[i, :, :, 0].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)
        reduced_im[i, :, :, 1] = cv2.resize(im[i, :, :, 1].numpy(), dsize=(128, 128),
                                            interpolation=cv2.INTER_LINEAR)

    return transforms.to_tensor(reduced_im)


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
