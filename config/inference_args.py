from .base_args import base_options


def inference_options():
    parser = base_options()
    parser.add_argument("--df", type=str, help="dataframe name")
    parser.add_argument(
        "--output",
        type=str,
        default="alpha",
        help="Output to infere; one of alpha, ratio",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="If to do fold inference. fold=0 means there are no folds.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model Filename; if there are multiple files, the folder name containing those",
    )
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="max",
        help="How to convert heatmaps to coords",
    )

    return parser
