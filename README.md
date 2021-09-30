# MRIGAN
This repository exists as a framework through which we can rapidly prototype
various GAN configurations for our project. This code is still a work in progress
with results being added as they come in.

## Core Architecture
The GAN architecture consists of two networks, a discriminator and a generator.

In this code the generator is a U-Net [1]. The
standard in the literature where a U-Net is used as a generator [2]-[5] is to
use residual blocks at each resolution. Thus, this U-Net uses residual blocks
and the overall architecture implemented in [2] as a starting point.

The discriminator is also implemented based on [2] as a starting point.

## Currently Implemented
As of right now the different tr