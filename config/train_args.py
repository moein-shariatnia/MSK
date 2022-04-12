from numpy.core.numeric import flatnonzero
from .base_args import base_options


def train_options():
    parser = base_options()
    parser.add_argument("--df", type=str, help="dataframe name")
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number to train on: one of [0, 1, 2, 3, 4]",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="num workers to use in pytorch dataloader",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning Rate of the optimizer",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight Decay of the optimizer"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="OneCycle",
        help="One of ['ReduceLR', 'OneCycle', 'Cosine']",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/weights",
        help="Where to save model weights",
    )

    return parser
