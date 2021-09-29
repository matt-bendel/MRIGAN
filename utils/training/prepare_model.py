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


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cuda'))
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def resume_train(args):
    checkpoint, model, optimizer = load_model(args.checkpoint)
    args = checkpoint['args']
    best_dev_loss = checkpoint['best_dev_loss']
    start_epoch = checkpoint['epoch']
    del checkpoint
    return model, optimizer, args, best_dev_loss, start_epoch


def fresh_start(args):
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    optimizer = build_optim(args, model.parameters())
    best_dev_loss = 1e9
    start_epoch = 0
    return model, optimizer, best_dev_loss, start_epoch
