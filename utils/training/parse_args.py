import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    # PATCH ARGS
    parser.add_argument('--patches', action='store_true', help='Whether or not only loss is adversarial')
    parser.add_argument('--num-patches', type=int, default=4, help='Number of z values')

    # ABLATION ARGS
    parser.add_argument('--adv-only', action='store_true', help='Whether or not only loss is adversarial')
    parser.add_argument('--supervised', action='store_true', help='Whether or not to use supervised loss')
    parser.add_argument('--var-loss', action='store_true', help='Whether or not to use variation loss')
    parser.add_argument('--data-consistency', action='store_true', help='Whether or not to use data consistency')

    # GAN ARGS
    parser.add_argument('--network-input', type=str, required=True, help='Image or K-Space U-Net')
    parser.add_argument('--disc-kspace', action='store_true', help='Image or K-Space Discriminator')
    parser.add_argument('--mbsd', action='store_true', help='Whether or not to use mbsd')
    parser.add_argument('--num-iters-discriminator', type=int, default=1,
                        help='Number of iterations of the discriminator')
    parser.add_argument('--num-z', type=int, default=8,
                        help='Number of z values')
    parser.add_argument('--latent-size', type=int, default=512, help='Size of latent vector for z location 2')
    parser.add_argument('--z-location', type=int, required=True, help='Where to put code vector')

    # LEARNING ARGS
    parser.add_argument('--batch-size', default=4, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0, help='Beta 1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='Beta 2 for Adam')

    # DATA ARGS
    parser.add_argument('--im-size', default=384, type=int,
                        help='Image resolution')
    parser.add_argument('--data-parallel', required=True, action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--adler', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--noise-v-fjd', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--num_of_top_slices', default=6, type=int,
                        help='top slices have bigger brain image and less air region')
    parser.add_argument('--use-middle-slices', action='store_true',
                        help='If set, only uses central slice of every data collection')

    # LOGISTICAL ARGS
    parser.add_argument('--report-interval', type=int, default=5, help='Period of loss reporting')
    parser.add_argument('--inpaint', action='store_true', help='Whether or not to remove chunk of image')
    parser.add_argument('--dynamic-inpaint', action='store_true',
                        help='Whether or not to dynamically remove chunk of image')
    parser.add_argument('--device', type=int, default=0,
                        help='Which device to train on. Use idx of cuda device or -1 for CPU')
    parser.add_argument('--exp-dir', type=pathlib.Path,
                        default=pathlib.Path('/home/bendel.8/Git_Repos/full_scale_mrigan/MRIGAN/trained_models'),
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint_gen', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--checkpoint_dis', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    return parser
