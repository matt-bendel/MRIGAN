import torch

import numpy as np
import sigpy as sp
from espirit import ifft, fft
from torch.utils.data import DataLoader
from data import transforms
from utils.math import complex_abs
from data.mri_data_mvue import SelectiveSliceData, SelectiveSliceData_Val
from utils.fftc import ifft2c_new, fft2c_new
import cv2
import torchgeometry as tg

def get_mvue(kspace, s_maps):
    ''' Get mvue estimate from coil measurements '''
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(np.sum(np.square(np.abs(s_maps)), axis=1))

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, val, use_seed=False):
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
        self.val = val
        self.mask = None
        self.image_size = (384, 384)

    def _get_mask(self, acs_lines=30, total_lines=384, R=1, pattern='random'):
        # Create a mask and place ones at the right locations
        a = np.array(
            [0, 10, 19, 28, 37, 46, 54, 61, 69, 76, 83, 89, 95, 101, 107, 112, 118, 122, 127, 132, 136, 140, 144, 148,
             151, 155, 158, 161, 164,
             167, 170, 173, 176, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 214, 217, 220,
             223, 226, 229, 233, 236,
             240, 244, 248, 252, 257, 262, 266, 272, 277, 283, 289, 295, 301, 308, 315, 323, 330, 338, 347, 356, 365,
             374])
        mask = np.zeros(384)
        mask[a] = 1
        mask[176:208] = 1

        return mask

    def __call__(self, kspace, target, attrs, fname, slice, sense_maps):
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
        gt_ksp = kspace

        # Crop extra lines and reduce FoV in phase-encode
        gt_ksp = sp.resize(gt_ksp, (
            gt_ksp.shape[0], gt_ksp.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size[0],
                                    gt_ksp.shape[2]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,))  # Back to k-space

        # Crop extra lines and reduce FoV in phase-encode
        maps = sense_maps
        maps = sp.fft(maps, axes=(-2, -1))  # These are now maps in k-space
        maps = sp.resize(maps, (
            maps.shape[0], maps.shape[1], self.image_size[1]))

        # Reduce FoV by half in the readout direction
        maps = sp.ifft(maps, axes=(-2,))
        maps = sp.resize(maps, (maps.shape[0], self.image_size[0],
                                maps.shape[2]))
        maps = sp.fft(maps, axes=(-2,))  # Back to k-space
        maps = sp.ifft(maps, axes=(-2, -1))  # Finally convert back to image domain

        # find mvue image
        gt = torch.tensor(get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps.reshape((1,) + maps.shape)))[0].abs().unsqueeze(0).repeat(3, 1, 1)

        mask = self._get_mask(None, None,
                              4, None)
        # Mask k-space
        gt_ksp *= mask[None, None, :]

        zfr = torch.tensor(get_mvue(gt_ksp, maps))[0].abs().unsqueeze(0).repeat(3, 1, 1)

        ref_im = 2*(gt - torch.min(gt))/(torch.max(gt) - torch.min(gt)) - 1
        cond_im = 2*(torch.clone(zfr) - torch.min(zfr))/(torch.max(zfr) - torch.min(zfr)) - 1

        if self.args.patches and self.args.num_patches > 1:
            new_cond_im = torch.zeros(self.args.num_patches**2, 3, 384 // (self.args.num_patches), 384 // (self.args.num_patches))
            new_ref_im = torch.zeros(self.args.num_patches**2, 3, 384 // (self.args.num_patches), 384 // (self.args.num_patches))
            col = 0
            for i in range(self.args.num_patches**2):
                ind = i % self.args.num_patches
                if i < self.args.num_patches and i % self.args.num_patches == 0:
                    col = 0
                elif i % self.args.num_patches == 0:
                    col += 384 // self.args.num_patches

                new_cond_im[i, :, :, :] = cond_im[:, ind*384//(self.args.num_patches):(ind+1)*384//(self.args.num_patches), col:col+384//(self.args.num_patches)]
                new_ref_im[i, :, :, :] = ref_im[:, ind*384//(self.args.num_patches):(ind+1)*384//(self.args.num_patches), col:col+384//(self.args.num_patches)]

            ref_im = new_ref_im
            cond_im = new_cond_im


        return cond_im, ref_im


def create_datasets(args, val_only):
    train_data = SelectiveSliceData(
        root=args.data_path / 'multicoil_train',
        transform=DataTransform(args, val=False),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
    )

    dev_data = SelectiveSliceData_Val(
        root=args.data_path / 'multicoil_val',
        transform=DataTransform(args, val=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
        big_test=True
    )

    return dev_data, train_data

def create_small_dataset(args, val_only):
    dev_data = SelectiveSliceData_Val(
        root=args.data_path / 'multicoil_val',
        transform=DataTransform(args, val=True),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        restrict_size=False,
        big_test=True
    )

    return dev_data

def create_data_loaders(args, val_only=False):
    dev_data, train_data = create_datasets(args, val_only)

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

    return train_loader, dev_loader


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
