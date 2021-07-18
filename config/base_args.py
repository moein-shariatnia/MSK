import argparse


def base_options():
    parser = argparse.ArgumentParser(description="Base Arguments")
    parser.add_argument(
        "--mode", type=str, default="train", help="Wheter 'train' or 'valid/test' model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset",
        help="Path to data directory",
    )
    parser.add_argument("--data_subfolder", type=str, default="train", help="Dataset subfolder: one of train, valid")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--point_count", type=int, default=3, help="How many points to predict")
    parser.add_argument("--coefficient", type=int, default=10, help="Coefficient for heatmap generation")
    parser.add_argument("--model", type=str, default="resnet18", help="Model to use")
    parser.add_argument("--heatmap", type=str, default="gaussian", help="Type of heatmap to use")
    parser.add_argument("--intensity", type=str, default="no_aug", help="How much data augmentation to use")


    return parser
