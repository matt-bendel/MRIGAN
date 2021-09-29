import torch

from models.generator.generator import GeneratorModel
from models.discriminator.discriminator import DiscriminatorModel


def build_model(args):
    model = GeneratorModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        z_location=args.z_location,
        model_type=args.network_input
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


def load_model(checkpoint_file_gen, checkpoint_file_dis):
    checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))
    checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

    args = checkpoint_gen['args']
    generator = build_model(checkpoint_gen['args'])
    discriminator = build_discriminator(checkpoint_dis['args'])

    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    generator.load_state_dict(checkpoint_gen['model'])
    discriminator.load_state_dict(checkpoint_dis['model'])

    return checkpoint_gen, generator, checkpoint_dis, discriminator


def resume_train(args):
    checkpoint_gen, generator, checkpoint_dis, discriminator = load_model(args.checkpoint_gen, args.checkpoint_dis)
    args = checkpoint_gen['args']
    best_dev_loss = checkpoint_gen['best_dev_loss']
    start_epoch = checkpoint_gen['epoch']
    del checkpoint_gen
    del checkpoint_dis
    return generator, discriminator, args, best_dev_loss, start_epoch


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
