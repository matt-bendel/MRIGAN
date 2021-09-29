import pathlib

from utils.args import Args

def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # MODEL SPECIFIC ARGS
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=192, help='Number of U-Net channels')
    parser.add_argument('--network-input', type=str, required=True, help='Image or K-Space U-Net')

    # GAN ARGS
    parser.add_argument('--num-iters-discriminator', type=int, default=3, help='Number of iterations of the discriminator')
    parser.add_argument('--z-location', type=int, required=True, help='Where to put code vector')

    # LEARNING ARGS
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of training epochs')

    # DATA ARGS
    parser.add_argument('--data-parallel', required=True, action='store_true', help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num_of_top_slices', default=6, type=int, help='top slices have bigger brain image and less air region')
    parser.add_argument('--use-middle-slices', action='store_true', help='If set, only uses central slice of every data collection')

    # LOGISTICAL ARGS
    parser.add_argument('--report-interval', type=int, default=5, help='Period of loss reporting')
    parser.add_argument('--device', type=int, default=0, help='Which device to train on. Use idx of cuda device or -1 for CPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, required=True, help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')


    return parser