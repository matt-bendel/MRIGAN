import torch

import numpy as np

from espirit import ifft, fft
from torch.utils.data import DataLoader
from data import transforms

from data.mri_data import SelectiveSliceData, SelectiveSliceData_Val


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
        m = np.zeros((384, 384))
        m[:, a] = True
        m[:, 176:208] = True
        samp = m
        numcoil = 8
        mask = np.tile(samp, (numcoil, 1, 1)).transpose((1, 2, 0)).astype(np.float32)

        kspace = kspace.transpose(1, 2, 0)

        x = ifft(kspace, (0, 1))  # (768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x)  # (384, 384, 8)

        kspace = fft(coil_compressed_x, (1, 0))  # (384, 384, 8)

        masked_kspace = kspace * mask

        kspace = transforms.to_tensor(kspace)
        kspace = kspace.permute(2, 0, 1, 3)

        masked_kspace = transforms.to_tensor(masked_kspace)
        masked_kspace = masked_kspace.permute(2, 0, 1, 3)

        # Apply mask
        nnz_index_mask = mask[0, :, 0].nonzero()[0]

        noise_var = torch.tensor(5.3459594390181664e-11)

        nnz_masked_kspace = masked_kspace[:, :, nnz_index_mask, :]
        nnz_masked_kspace_real = nnz_masked_kspace[:, :, :, 0]
        nnz_masked_kspace_imag = nnz_masked_kspace[:, :, :, 1]
        nnz_masked_kspace_real_flat = flatten(nnz_masked_kspace_real)
        nnz_masked_kspace_imag_flat = flatten(nnz_masked_kspace_imag)

        noise_flat_1 = (torch.sqrt(0.5 * noise_var)) * torch.randn(nnz_masked_kspace_real_flat.size())
        noise_flat_2 = (torch.sqrt(0.5 * noise_var)) * torch.randn(nnz_masked_kspace_real_flat.size())

        nnz_masked_kspace_real_flat_noisy = nnz_masked_kspace_real_flat.float() + noise_flat_1
        nnz_masked_kspace_imag_flat_noisy = nnz_masked_kspace_imag_flat.float() + noise_flat_2

        nnz_masked_kspace_real_noisy = unflatten(nnz_masked_kspace_real_flat_noisy, nnz_masked_kspace_real.shape)
        nnz_masked_kspace_imag_noisy = unflatten(nnz_masked_kspace_imag_flat_noisy, nnz_masked_kspace_imag.shape)

        nnz_masked_kspace_noisy = nnz_masked_kspace * 0
        nnz_masked_kspace_noisy[:, :, :, 0] = nnz_masked_kspace_real_noisy
        nnz_masked_kspace_noisy[:, :, :, 1] = nnz_masked_kspace_imag_noisy

        masked_kspace_noisy = 0 * masked_kspace
        masked_kspace_noisy[:, :, nnz_index_mask, :] = nnz_masked_kspace_noisy

        ## commenting the bellow one line will make the experiment noiseless case
        # masked_kspace = masked_kspace_noisy

        ###################################

        if self.args.z_location == 3:
            stacked_masked_kspace = torch.zeros(17, 384, 384)
        else:
            stacked_masked_kspace = torch.zeros(16, 384, 384)

        stacked_masked_kspace[0:8, :, :] = torch.squeeze(masked_kspace[:, :, :, 0])
        stacked_masked_kspace[8:16, :, :] = torch.squeeze(masked_kspace[:, :, :, 1])
        stacked_masked_kspace, mean, std = transforms.normalize_instance(stacked_masked_kspace, eps=1e-11)
        # stacked_masked_kspace = (stacked_masked_kspace - (-4.0156e-11)) / (2.5036e-05)

        stacked_kspace = torch.zeros(16, 384, 384)
        stacked_kspace[0:8, :, :] = torch.squeeze(kspace[:, :, :, 0])
        stacked_kspace[8:16, :, :] = torch.squeeze(kspace[:, :, :, 1])
        stacked_kspace = transforms.normalize(stacked_kspace, mean, std, eps=1e-11)
        # stacked_kspace = (stacked_kspace - (-4.0156e-11)) / (2.5036e-05)

        # mean = (-4.0156e-11)
        # std = (2.5036e-05)

        return stacked_masked_kspace, stacked_kspace, mean, std, nnz_index_mask


def create_datasets(args):
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

    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

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