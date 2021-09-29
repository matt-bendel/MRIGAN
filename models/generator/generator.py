"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NOTE: z_location tells the network where to use the latent variable. It has options:
    0: No latent vector
    1: Add latent vector to zero filled areas
    2: Add latent vector to middle of network (between encoder and decoder)
    3: Add as an extra input channel
"""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, batch_norm=True, down=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.batch_norm = batch_norm

        if self.in_chans != self.out_chans:
            self.out_chans = self.in_chans

        self.norm = nn.BatchNorm2d(self.out_chans)
        self.conv_1_x_1 = nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(1, 1))
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input
        if self.batch_norm:
            output = self.norm(input)

        return self.layers(output) + self.conv_1_x_1(output)


class FullDownBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.resblock = ResidualBlock(self.out_chans, self.out_chans, True)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        return self.resblock(self.downsample(input))

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_chans}, out_chans={self.out_chans}\nResBlock(in_chans={self.out_chans}, out_chans={self.out_chans}'


class FullUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.upsample = nn.Sequential(
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.resblock = ResidualBlock(self.out_chans*2, self.out_chans*2, True, down=False)

    def forward(self, input, old):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = self.upsample(input)
        output = torch.cat([self.upsample(output), old], dim=1)

        return self.resblock(output)

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_chans}, out_chans={self.out_chans}\nResBlock(in_chans={self.out_chans}, out_chans={self.out_chans}'


class GeneratorModel(nn.Module):
    def __init__(self, in_chans, out_chans, z_location, model_type):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.z_location = z_location
        self.model_type = model_type

        if z_location == 3:
            self.in_chans = in_chans + 1

        self.initial_layers = nn.Sequential(
            nn.Conv2d(self.in_chans, 32, kernel_size=(3, 3), padding=1),  # 384x384
            ResidualBlock(32, 32, False),
        )

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers += [FullDownBlock(32, 64)]  # 192x192
        self.encoder_layers += [FullDownBlock(64, 128)]  # 96x96
        self.encoder_layers += [FullDownBlock(128, 256)]  # 48x48
        self.encoder_layers += [FullDownBlock(256, 512)]  # 24x24
        self.encoder_layers += [FullDownBlock(512, 512)]  # 12x12
        self.encoder_layers += [FullDownBlock(512, 512)]  # 6x6

        if z_location == 2:
            # WE CHANGE THE MIDDLE OF THE MODEL
            print(2)
        else:
            self.middle = ResidualBlock(512, 512)  # 6x6

        self.decoder_layers = nn.ModuleList()
        self.decoder_layers += [FullUpBlock(512, 512)]  # 12x12
        self.decoder_layers += [FullUpBlock(512, 512)]  # 24x24
        self.decoder_layers += [FullUpBlock(512, 256)]  # 48x48
        self.decoder_layers += [FullUpBlock(256, 128)]  # 96x96
        self.decoder_layers += [FullUpBlock(128, 64)]  # 192x192
        self.decoder_layers += [FullUpBlock(64, 32)]  # 384x384

        self.final_conv = nn.Conv2d(32, 16, kernel_size=(1, 1))

    def forward(self, input, z=None):
        output = input
        output = self.initial_layers(output)

        stack = []
        stack.append(output)

        # Apply down-sampling layers
        for layer in self.encoder_layers:
            output = layer(output)
            stack.append(output)

        stack.pop()
        if self.z_location == 2:
            output = self.middle(output, z)
        else:
            output = self.middle(output)

        # Apply up-sampling layers
        for layer in self.decoder_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = layer(output, stack.pop())

        return self.final_conv(output)
