import torch

import numpy as np

from espirit import ifft, fft
from torch.utils.data import DataLoader
from data import transforms
from utils.math import complex_abs
from data.mri_data import SelectiveSliceData, SelectiveSliceData_Val
from utils.fftc import ifft2c_new, fft2c_new
import cv2


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
        # TODO: ADD LOGIC FOR 384 IMS
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

        im_tensor = reduce_resolution(complex_coil_im) if self.args.im_size != 384 else complex_coil_im

        if self.args.inpaint:
            im_tensor = transforms.root_sum_of_squares(complex_abs(im_tensor)).unsqueeze(0)

            input_tensor = torch.clone(im_tensor)

            n = input_tensor.shape[1]
            square_length = n // 3
            end = n - square_length

            # rand_start_col = randrange(0, end)
            # rand_start_row = randrange(0, end)
            rand_start_row = (n - square_length) // 2
            rand_start_col = (n - square_length) // 2

            input_tensor[:, rand_start_row:rand_start_row + square_length,
            rand_start_col:rand_start_col + square_length] = 0
            # input_tensor[:, :, 64:128] = 0

            normalized_input, mean, std = transforms.normalize_instance(input_tensor)
            normalized_gt = transforms.normalize(im_tensor, mean, std)

            ref_im = normalized_gt.repeat(3, 1, 1)
            cond_im = normalized_input.repeat(3, 1, 1)

            ref_im  = 2 * (ref_im - torch.min(ref_im)) / (torch.max(ref_im) - torch.min(ref_im)) - 1
            cond_im = 2 * (torch.clone(cond_im) - torch.min(cond_im)) / (torch.max(cond_im) - torch.min(cond_im)) - 1

            return normalized_input, normalized_gt, cond_im, ref_im

        true_image = torch.clone(im_tensor)
        image = im_tensor

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

        ref_im = torch.zeros(8, 128, 128, 2)
        ref_im[:, :, :, 0] = stacked_image[0:8, :, :]
        ref_im[:, :, :, 1] = stacked_image[8:16, :, :]
        ref_im = transforms.root_sum_of_squares(complex_abs(ref_im)).unsqueeze(0).repeat(3, 1, 1)

        cond_im = torch.zeros(8, 128, 128, 2)
        cond_im[:, :, :, 0] = stacked_masked_zfr[0:8, :, :]
        cond_im[:, :, :, 1] = stacked_masked_zfr[8:16, :, :]
        cond_im = transforms.root_sum_of_squares(complex_abs(cond_im)).unsqueeze(0).repeat(3, 1, 1)

        ref_im = 2*(ref_im - torch.min(ref_im))/(torch.max(ref_im) - torch.min(ref_im)) - 1
        cond_im = 2*(torch.clone(cond_im) - torch.min(cond_im))/(torch.max(cond_im) - torch.min(cond_im)) - 1

        if self.args.patches:
            new_cond_im = torch.zeros(self.args.num_patches, 3, 128 // self.args.num_patches**2, 128 // self.args.num_patches**2)
            new_ref_im = torch.zeros(self.args.num_patches, 3, 128 // self.args.num_patches**2, 128 // self.args.num_patches**2)
            col = 0
            for i in range(self.args.num_patches**2):
                ind = i % self.args.num_patches
                if i < self.args.num_patches and i % self.args.num_patches == 0:
                    col = 0
                elif i % self.args.num_patches == 0:
                    col += 128 // self.args.num_patches

                new_cond_im[i, :, :, :] = cond_im[:, ind*128//self.args.num_patches:(ind+1)*128//self.args.num_patches, col:col+128//self.args.num_patches]
                new_ref_im[i, :, :, :] = ref_im[:, ind*128//self.args.num_patches:(ind+1)*128//self.args.num_patches, col:col+128//self.args.num_patches]


        return stacked_masked_zfr, cond_im, ref_im


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
