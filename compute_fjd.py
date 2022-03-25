# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
'''
How to Use FJD

This script provides a simple example of how to evaluate a conditional GAN using FJD.
'''
import pathlib

import torch
import matplotlib.pyplot as plt
import numpy as np
import fjd_metric_langevin
import cfid_tests

from utils.general.helper import readd_measures_im
from utils.training.prepare_data_fid import create_data_loaders
from utils.training import prepare_data_fid_langevin
from utils.training.parse_args import create_arg_parser
from fjd_metric import FJDMetric
from embeddings import InceptionEmbedding
from data import transforms
from utils.math import complex_abs

'''
In order to compute FJD we will need two data loaders: one to provide images and 
conditioning for the reference distribution, and a second one whose conditioning 
will be used to condition the GAN for creating the generated distribution. For 
this example we will use the CIFAR-10 dataset.

When loading in reference images, it is important to normalize them between [-1, 1].
'''


def get_dataloaders(args):
    train_loader, test_loader = create_data_loaders(args)

    return train_loader, test_loader


def get_gen(args):
    from utils.training.prepare_model import build_model, build_optim, build_discriminator
    string_for_file = '/ablation'  # if args.ablation else '/'
    if args.inpaint:
        checkpoint_file_gen = pathlib.Path(
            f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/inpaint/generator_best_model.pt')
    else:
        checkpoint_file_gen = pathlib.Path(
            f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models{string_for_file}/image/{args.z_location}/generator_best_model.pt')

    if args.adler and args.z_location != 7:
        checkpoint_file_gen = pathlib.Path(
            f'/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models/adler/generator_best_model.pt')

    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

    generator = build_model(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    generator.load_state_dict(checkpoint_gen['model'])

    return generator


'''
In order to be able to accomodate a wide variety of model configurations, we 
use a GAN wrapper to standardize model inputs and outputs. Each model is 
expected to take as input a set of conditions y, and return a corresponding 
set of generated samples.
'''


class GANWrapper:
    def __init__(self, model, args, noise_var=None):
        self.args = args
        self.model = model
        self.device = args.device
        self.noise_var = noise_var
        if not args.noise_v_fjd:
            self.model.eval()
        self.data_consistency = True if args.z_location != 5 else False

    def get_noise(self, batch_size):
        # change the noise dimension as required
        if not self.args.adler:
            z = torch.cuda.FloatTensor(
                np.random.normal(size=(batch_size, 512), scale=np.sqrt(1)))
        else:
            z = torch.rand((batch_size, 2, 128, 128)).cuda()

        return z

    def convert_to_im(self, samples):
        if self.args.inpaint:
            temp = samples.squeeze(1)
        else:
            temp = torch.zeros((samples.size(0), 8, 128, 128, 2)).to(self.device)
            temp[:, :, :, :, 0] = samples[:, 0:8, :, :]
            temp[:, :, :, :, 1] = samples[:, 8:16, :, :]

        final_im = torch.zeros(size=(samples.size(0), 3, 128, 128)).to(self.device)
        for i in range(samples.size(0)):
            im = transforms.root_sum_of_squares(complex_abs(temp[i])) if not self.args.inpaint else temp[i]
            im = 2 * (im - torch.min(im)) / (torch.max(im) - torch.min(im)) - 1
            final_im[i, 0, :, :] = im
            final_im[i, 1, :, :] = im
            final_im[i, 2, :, :] = im

        return final_im

    def patch_im(self, im):
        new_im = torch.zeros(im.size(0), self.args.num_patches ** 2, 3, 128 // (self.args.num_patches),
                             128 // (self.args.num_patches))
        col = 0
        for i in range(self.args.num_patches ** 2):
            ind = i % self.args.num_patches
            if i < self.args.num_patches and i % self.args.num_patches == 0:
                col = 0
            elif i % self.args.num_patches == 0:
                col += 128 // self.args.num_patches

            new_im[:, i, :, :, :] = im[:, :,
                                    ind * 128 // (self.args.num_patches):(ind + 1) * 128 // (self.args.num_patches),
                                    col:col + 128 // (self.args.num_patches)]

        return new_im

    def add_noise(self, im):
        return im + torch.empty((im.size(0), 3, 128, 128)).normal_(mean=0, std=self.noise_var).cuda()

    def free_memory(self):
        del self.model

    def __call__(self, y, target=None):
        if self.args.noise_v_fjd:
            return self.add_noise(y)

        batch_size = y.size(0)
        inds = torch.nonzero(y == 0) if self.args.inpaint else None
        z = self.get_noise(batch_size)
        samples = self.model(y, z) if not self.args.adler else self.model(torch.cat([y, z], dim=1))
        if self.args.inpaint:
            samples[inds] = target[inds]
            im = self.convert_to_im(samples)
        else:
            samples = readd_measures_im(samples, y, args, true_measures=y) if self.data_consistency else samples
            im = self.convert_to_im(samples)
        return im if not self.args.patches else self.patch_im(im)


'''
The FJDMetric object handles embedding the images and conditioning, the 
computation of the reference distribution and generated distribution statistics, 
the scaling of the conditioning component with alpha, and the calculation of FJD. 

It requires several inputs:

gan - A GAN model which takes as input conditioning and yields image samples as 
    output.
reference_loader - A data loader for the reference distribution, which yields 
    image-condition pairs.
condition_loader - A data loader for the generated distribution, which yields 
    image-condition pairs. Images are ignored, and the conditioning is used as 
    input to the GAN.
image_embedding - An image embedding function. This will almost always be the 
    InceptionEmbedding.
condition_embedding - A conditioning embedding function. As we are dealing with 
    class conditioning in this example, we will use one-hot encoding.

Other options:

save_reference_stats - Indicates whether the statistics of the reference 
    distribution should be saved to the path provided in reference_stats_path. 
    This can speed up computation of FJD if the same reference set is used for 
    multiple evaluations.
samples_per_condition - Indicates the number of images that will be generated 
    for each condition drawn from the condition loader. This may be useful if 
    there are very few samples in the conditioning dataset, or to emphasize 
    intra-conditioning diversity when calculating FJD.
cuda - If True, indicates that the GPU accelerated version of FJD should be 
    used. This version should be considerably faster than the CPU version, but 
    may be slightly more unstable.
'''


def main(args):
    print("GETTING DATA LOADERS")
    print("GETTING INCEPTION EMBEDDING")
    inception_embedding = InceptionEmbedding(parallel=True)
    print("GETTING GENERATOR")
    max = 6 if not args.inpaint and not args.adler else 1
    Langevin = False
    cfid_test = False

    if cfid_test:
        args.patches = False
        ref_loader, cond_loader = create_data_loaders(args)

        args.z_location = 1
        gan = get_gen(args)
        gan = GANWrapper(gan, args)
        print("COMPUTING METRIC")
        fjd_metric = FJDMetric(gan=gan,
                               reference_loader=ref_loader,
                               condition_loader=cond_loader,
                               image_embedding=inception_embedding,
                               condition_embedding=inception_embedding,
                               reference_stats_path=f'ref_stats_{args.num_patches}.npz' if args.patches else 'ref_stats.npz',
                               save_reference_stats=True,
                               samples_per_condition=8,
                               cuda=True,
                               args=args)

        print(f"FID FOR NETWORK {args.z_location}")
        fid = fjd_metric.get_fid()
        fjd = fjd_metric.get_fjd(alpha=1.097)
        print('FID: ', fid)
        print('FJD: ', fjd)
        gan.free_memory()
        del gan
        del fjd_metric.image_embedding
        del fjd_metric.condition_embedding
        del fjd_metric.reference_loader
        del fjd_metric.condition_loader
        del fjd_metric.gan
        cfid_val, cov1 = fjd_metric.get_cfid_torch()
        cfid_val_tf, cov2 = fjd_metric.get_cfid()

        print('CFID-Torch: ', cfid_val)
        print('CFID-Tensorflow: ', cfid_val_tf)

    if Langevin:
        print("COMPUTING METRIC")
        args.patches = True
        for i in range(5):
            args.num_patches = 2 ** i
            print("PATCHES ", args.num_patches ** 2)
            ref_loader, cond_loader = prepare_data_fid_langevin.create_data_loaders(args)
            fjd_metric = fjd_metric_langevin.FJDMetric(gan=None,
                                                       reference_loader=ref_loader,
                                                       condition_loader=cond_loader,
                                                       image_embedding=inception_embedding,
                                                       condition_embedding=inception_embedding,
                                                       reference_stats_path=f'ref_stats_mvue_{args.num_patches}.npz',
                                                       save_reference_stats=True,
                                                       samples_per_condition=32,
                                                       cuda=True,
                                                       args=args)
            fid = fjd_metric.get_fid()
            fjd = fjd_metric.get_fjd(alpha=1.097)
            print('FID: ', fid)
            print('FJD: ', fjd)
            cfid_val = fjd_metric.get_cfid()
            print('CFID: ', cfid_val)
            del fjd_metric
            del ref_loader
            del cond_loader
        exit()

    if args.patches and not args.inpaint:
        num_samps = 1
        for i in range(3):
            args.num_patches = 2**i
            if args.num_patches == 1:
                args.patches = False
            else:
                args.patches = True
            print("PATCHES ", args.num_patches)
            ref_loader, cond_loader = get_dataloaders(args)
            for j in range(9):
                # if j == 0 or j == 5 or j == 6:
                #     continue
                args.z_location = j+1
                args.adler = True if j > 5 and j != 8 else False
                if j == 8:
                    args.inpaint = True
                    args.in_chans = 1
                    args.out_chans = 1
                else:
                    args.inpaint = False
                    args.in_chans = 16
                    args.out_chans = 16
                gan = get_gen(args)
                gan = GANWrapper(gan, args)
                print("COMPUTING METRIC")
                fjd_metric = FJDMetric(gan=gan,
                                       reference_loader=ref_loader,
                                       condition_loader=cond_loader,
                                       image_embedding=inception_embedding,
                                       condition_embedding=inception_embedding,
                                       reference_stats_path=f'ref_stats_{args.num_patches}.npz' if args.patches else 'ref_stats.npz',
                                       save_reference_stats=True,
                                       samples_per_condition=num_samps,
                                       cuda=True,
                                       args=args)

                print(f"FID FOR NETWORK {args.z_location}")
                fid = fjd_metric.get_fid()
                fjd = fjd_metric.get_fjd(alpha=1.097)
                print('FID: ', fid)
                print('FJD: ', fjd)
                gan.free_memory()
                del gan
                del fjd_metric.image_embedding
                del fjd_metric.condition_embedding
                del fjd_metric.reference_loader
                del fjd_metric.condition_loader
                del fjd_metric.gan
                cfid_val = fjd_metric.get_cfid_torch()
                print('CFID: ', cfid_val)
                del fjd_metric
            del ref_loader
            del cond_loader
        exit()
    else:
        ref_loader, cond_loader = get_dataloaders(args)

    if args.noise_v_fjd:
        for i in range(6):
            power = i - 5
            exponent = 10 ** power
            gan = None
            gan = GANWrapper(gan, args, noise_var=exponent)
            print("COMPUTING METRIC")
            fjd_metric = FJDMetric(gan=gan,
                                   reference_loader=ref_loader,
                                   condition_loader=cond_loader,
                                   image_embedding=inception_embedding,
                                   condition_embedding=inception_embedding,
                                   save_reference_stats=True,
                                   samples_per_condition=128,
                                   cuda=True,
                                   args=args)

            '''
                    Once the FJD object is initialized, FID and FJD can be calculated by calling 
                    get_fid or get_fjd. By default, the alpha value used to weight the 
                    conditional component of FJD is selected to be the ratio between the 
                    average L2 norm of the image embedding and conditioning embedding.

                    We see in this example that even though our "GAN" gets a very good FID 
                    score due to the generated image distribution being very close to the 
                    reference image distribution, its FJD score is very bad, as the model lacks 
                    any conditional consistency.
                    '''
            print(f"FID FOR NOISE STD. DEV: {exponent}")
            fid = fjd_metric.get_fid()
            fjd = fjd_metric.get_fjd(alpha=1.097)
            print('FID: ', fid)
            print('FJD: ', fjd)
            gan.free_memory()
            del gan
            del fjd_metric

    else:
        for i in range(8):
            num_samps = 32
            args.z_location = i + 1
            args.inpaint = True
            gan = get_gen(args)
            gan = GANWrapper(gan, args)
            print("COMPUTING METRIC")
            fjd_metric = FJDMetric(gan=gan,
                                   reference_loader=ref_loader,
                                   condition_loader=cond_loader,
                                   image_embedding=inception_embedding,
                                   condition_embedding=inception_embedding,
                                   save_reference_stats=True,
                                   reference_stats_path='ref_stats_inpaint.npz',
                                   samples_per_condition=num_samps,
                                   cuda=True,
                                   args=args)

            '''
                    Once the FJD object is initialized, FID and FJD can be calculated by calling 
                    get_fid or get_fjd. By default, the alpha value used to weight the 
                    conditional component of FJD is selected to be the ratio between the 
                    average L2 norm of the image embedding and conditioning embedding.

                    We see in this example that even though our "GAN" gets a very good FID 
                    score due to the generated image distribution being very close to the 
                    reference image distribution, its FJD score is very bad, as the model lacks 
                    any conditional consistency.
                    '''
            print(f"FID FOR NETWORK {args.z_location}")
            fid = fjd_metric.get_fid()
            fjd = fjd_metric.get_fjd(alpha=1.097)
            print('FID: ', fid)
            print('FJD: ', fjd)
            cfid_val = fjd_metric.get_cfid_torch()
            print('CFID: ', cfid_val)
            gan.free_memory()
            del gan
            del fjd_metric

    '''
    To visualize how FJD changes as we increase the weighting on the conditional 
    component, we can evaluate it at a range of alpha values using the 
    sweep_alpha function.
    '''
    # alpha = fjd_metric.alpha
    # alphas = [0, 1, 2, 4, 8, 16, 32]
    # fjds = fjd_metric.sweep_alpha(alphas)
    #
    # plt.plot(alphas, fjds, label='FJD', linewidth=3)
    # plt.plot(alphas, [fid] * len(alphas), label='FID', linewidth=3)
    # plt.axvline(x=alpha, c='black', label=r'Suggested $\alpha$', linewidth=2)
    # plt.xlabel(r'$\alpha$')
    # plt.ylabel('Distance')
    # plt.legend()
    # plt.savefig('fjd_alphas')
    # print(alpha)


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    args.in_chans = 16 if not args.inpaint else 1
    args.out_chans = 16 if not args.inpaint else 1
    main(args)
