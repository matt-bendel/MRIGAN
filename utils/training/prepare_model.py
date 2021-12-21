import torch

from models.generator.generator_experimental_2 import GeneratorModel
from models.discriminator.discriminator import DiscriminatorModel
from models.baseline_unet.unet_residual import UnetModelRes


def build_model(args):
    model = GeneratorModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        z_location=args.z_location,
        latent_size=512
    ).to(torch.device('cuda'))
    return model


def build_discriminator(args):
    model = DiscriminatorModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        z_location=args.z_location,
        model_type=args.network_input
    ).to(torch.device('cuda'))
    return model


def build_unet(args):
    model = UnetModelRes(
        in_chans=16,
        out_chans=16,
        chans=256,
        num_pool_layers=4,
    ).to(torch.device('cuda'))
    return model


def build_optim(args, params):
    return torch.optim.Adam(params, lr=args.lr, betas=(args.beta_1, args.beta_2))
    # return torch.optim.RMSprop(params, 0.0003)


def build_optim_unet(args, params):
    return torch.optim.RMSprop(params, 0.0003)


def load_model(checkpoint_file_gen, checkpoint_file_dis):
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))
    checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

    args = checkpoint_gen['args']
    generator = build_model(checkpoint_gen['args'])
    # discriminator = build_discriminator(checkpoint_dis['args'])

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)
        # discriminator = torch.nn.DataParallel(discriminator)

    generator.load_state_dict(checkpoint_gen['model'])
    # discriminator.load_state_dict(checkpoint_dis['model'])

    opt_gen = build_optim(args, generator.parameters())
    opt_gen.load_state_dict(checkpoint_gen['optimizer'])

    # opt_dis = build_optim(args, discriminator.parameters())
    # opt_dis.load_state_dict(checkpoint_dis['optimizer'])

    return checkpoint_gen, generator, opt_gen, checkpoint_dis, None, None


def load_model_unet(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))

    args = checkpoint['args']
    unet = build_unet(checkpoint['args'])

    if args.data_parallel:
        unet = torch.nn.DataParallel(unet)

    unet.load_state_dict(checkpoint['model'])

    opt = build_optim_unet(args, unet.parameters())
    opt.load_state_dict(checkpoint['optimizer'])

    return checkpoint, unet, opt


def resume_train(args):
    checkpoint_gen, generator, opt_gen, checkpoint_dis, discriminator, opt_dis = load_model(args.checkpoint_gen,
                                                                                            args.checkpoint_dis)
    args = checkpoint_gen['args']
    best_dev_loss = checkpoint_gen['best_dev_loss']
    start_epoch = checkpoint_gen['epoch']
    return generator, opt_gen, discriminator, opt_dis, args, best_dev_loss, start_epoch


def resume_train_unet(args):
    checkpoint, unet, opt = load_model_unet(args.checkpoint)
    args = checkpoint['args']
    best_dev_loss = checkpoint['best_dev_loss']
    start_epoch = checkpoint['epoch']
    return unet, opt, args, best_dev_loss, start_epoch


def fresh_start(args):
    generator = build_model(args)
    discriminator = build_discriminator(args)

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # We will use SSIM for dev loss
    best_dev_loss = 1e9
    start_epoch = 0
    return generator, discriminator, best_dev_loss, start_epoch
