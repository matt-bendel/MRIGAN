import torch

from models.unet.unet import UnetModel


def build_model(args):
    model = UnetModel(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        z_location=args.z_location,
        drop_prob=args.drop_prob,
        model_type=args.network_input
    ).to(torch.device('cuda'))
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    return checkpoint, model


def resume_train(args):
    checkpoint, model = load_model(args.checkpoint)
    args = checkpoint['args']
    best_dev_loss = checkpoint['best_dev_loss']
    start_epoch = checkpoint['epoch']
    del checkpoint
    return model, args, best_dev_loss, start_epoch


def fresh_start(args):
    generator = build_model(args)
    if args.data_parallel:
        generator = torch.nn.DataParallel(generator)

    #TODO: Implement and get discriminator model here
    best_dev_loss = 1e9
    start_epoch = 0
    return generator, generator, best_dev_loss, start_epoch
