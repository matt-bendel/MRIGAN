"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, norm=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.norm = norm

        if self.in_chans != self.out_chans:
            self.out_chans = self.in_chans

        # self.norm = nn.BatchNorm2d(self.out_chans)
        # self.conv_1_x_1 = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=(1, 1))
        self.layers = nn.Sequential(
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.out_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.out_chans, self.out_chans, kernel_size=(3, 3), padding=1),
        )

        self.final_act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input
        # if self.norm:
        #     output = self.norm(input)

        return self.final_act(torch.add(self.layers(output), input))


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.res = ResidualBlock(out_chans, out_chans, norm=norm)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.res(self.layers(input))

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
               f'drop_prob={self.drop_prob})'


class GeneratorModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, z_location, latent_size=None):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        chans = 32
        num_pool_layers = 5

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.z_location = z_location
        self.latent_size = latent_size

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, 0, norm=False)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, 0)]
            ch *= 2

        if z_location == 2:  # Concatenate z
            self.middle_z_grow_conv = nn.Sequential(
                nn.Conv2d(latent_size // 4, latent_size // 2, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(latent_size // 2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(latent_size // 2, latent_size, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(latent_size),
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.middle_z_grow_linear = nn.Sequential(
                nn.Linear(latent_size, latent_size // 4 * 3 * 3),
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.conv = nn.Sequential(
                nn.Conv2d(ch + latent_size, ch, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                ConvBlock(ch, ch, 0)
            )
        else:  # Add z if 2, add to all second half resolutions if 3
            self.conv = ConvBlock(ch, ch, 0)  # 6x6

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, 0)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, 0)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, input, z):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        if self.z_location == 2:
            z_out = self.middle_z_grow_linear(z)
            z_out = torch.reshape(z_out, (output.shape[0], self.latent_size // 4, 3, 3))
            z_out = self.middle_z_grow_conv(z_out)
            output = torch.cat([output, z_out], dim=1)
            output = self.conv(output)
        elif self.z_location == 3:
            output = self.conv(torch.add(output, z))
        else:
            output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)

        return self.conv2(output)
