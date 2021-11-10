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


def miniBatchStdDev(x, subGroupSize=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2] * size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3])).to(x.device)
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class ResidualBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, batch_norm=True):
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

        # self.norm = nn.BatchNorm2d(self.out_chans)
        self.conv_1_x_1 = nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(1, 1))
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(self.out_chans),
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
        # if self.batch_norm:
        #     output = self.norm(input)

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
            nn.InstanceNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
        )
        # self.resblock = ResidualBlock(self.out_chans, self.out_chans, True)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        return self.downsample(input)  # self.resblock(self.downsample(input))

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_chans}, out_chans={self.out_chans}\nResBlock(in_chans={self.out_chans}, out_chans={self.out_chans}'


class DiscriminatorModel(nn.Module):
    def __init__(self, in_chans, out_chans, z_location, model_type, mbsd=False):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_chans = 2
        self.out_chans = 2
        self.z_location = z_location
        self.model_type = model_type
        self.mbsd = mbsd

        # CHANGE BACK TO 16 FOR MORE
        self.initial_layers = nn.Sequential(
            nn.Conv2d(self.in_chans, 32, kernel_size=(3, 3), padding=1),  # 384x384
            nn.LeakyReLU()
        )

        self.encoder_layers = nn.ModuleList()
        # self.encoder_layers += [FullDownBlock(16, 32)]  # 192x192
        # self.encoder_layers += [FullDownBlock(32, 64)]  # 96x96
        self.encoder_layers += [FullDownBlock(32, 64)]  # 48x48
        self.encoder_layers += [FullDownBlock(64, 128)]  # 24x24
        self.encoder_layers += [FullDownBlock(128, 256)]  # 12x12
        self.encoder_layers += [FullDownBlock(256, 512)]  # 6x6
        self.encoder_layers += [FullDownBlock(512, 512)]  # 3x3

        self.post_mbsd_1 = nn.Sequential(
            nn.Conv2d(513, 513, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.post_mbsd_2 = nn.Sequential(
            nn.Conv2d(513, 513, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(513 * 3 * 3, 1),
        )

    def forward(self, input):
        output = self.initial_layers(input)

        # Apply down-sampling layers
        for layer in self.encoder_layers:
            output = layer(output)

        # COMMENT OUT MBSD WHEN NOT IN USE - ALSO IN INIT
        if self.mbsd:
            x = miniBatchStdDev(output)
            x = self.post_mbsd_1(x)
            # x = x.view(-1, num_flat_features(x))
            x = self.post_mbsd_2(x)

            output = x

        return self.dense(output)
