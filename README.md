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

See the paper for architecture specifics. The primary difference is the resolution
of the image at each step. We have input images of size 384x384 and
at the lowest resolution they are size 6x6. We use average pooling to downsample
and bilinear upsampling. We follow the same structure for the number of channels though.

In all, the network accepts an input of size (N, C, 384, 384) where N is the batch size and
C is the number of initial channels, 16 for most cases, but 17 for one (see below).

The motivation to have 16 input channels follows from the success of [6]. In that paper
both a k-space and image U-Net perform very well by using each coil as an input channel.
In our case, we are using T2 multicoil data from the fastMRI dataset. If the coils are compressable
and there are more than 8, they are compressed down to 8. Then, channels 1-8 are the real values of
coils 1-8 and channels 9-16 are the imaginary values of coils 1-8.

Note: The discriminator operates in the image space for **all** network variants. 

###Loss
This is a fully unsupervised task. We are using loss as described in WGAN-GP [7]. That is, we are using Wasserstein Distance on both
the discriminator and the generator in addition to a gradient penalty on the discriminator. See the
paper for details.

## The Goal
One should be able to train networks with the following configurations:
* K-space OR Image U-Net
    * z vector added as extra input channel
        * Input Size: (N, 17, 384, 384)
        * Total Trainable Params:
    * z vector added at unmeasured values
        * Input Size: (N, 16, 384, 384)
        * Total Trainable Params:
    * z vector concatenated at the bottom of the U-Net
        * Input Size: (N, 16, 384, 384)
        * Latent Input Size: z*N
        * Total Trainable Params:
    
### Currently Implemented
As of right now all three ways of injecting the latent variable
are implemented (in a very preliminary way) for the k-space U-Net.

### Not Yet Implemented
The image U-Net variants described above.

## Training a Network
In order to train a network, you may use the following templates.

### For z added to the unmeasured k-space

### For z as an extra input channel

### For z added at the bottom of the U-Net

## Testing
Not yet implemented.

## Dataset
The dataset used is the FastMRI dataset [8]. Specifically multicoil T2 brain images. 
It can be found [here](https://fastmri.med.nyu.edu/).

## References
NOTE: STILL NEEDS PROPERLY CITED

[1] https://arxiv.org/pdf/1505.04597.pdf

[2] https://arxiv.org/pdf/1811.05910.pdf

[3] https://www.sciencedirect.com/science/article/pii/S1361841518304596#fig0001

[4] https://arxiv.org/pdf/1511.06434.pdf

[5] https://arxiv.org/pdf/1505.04597.pdf

[6] https://arxiv.org/pdf/1910.12325.pdf

[7] https://arxiv.org/pdf/1704.00028.pdf

[8] https://arxiv.org/pdf/1811.08839.pdf