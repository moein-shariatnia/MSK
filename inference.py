import torch
import torch.nn as nn

import os
import time
from tqdm import tqdm
import pandas as pd

from models.model import UNet
from data.mura_dataset import MURADataset
from data.transforms import get_transforms
from config.inference_args import inference_options
from utils.metrics import get_coords, calculate_alpha_error, calculate_ratio_error


def build_model_dataloader(dataframe, options, model_filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(model=options.model, classes=options.point_count, pretrained=None).to(
        device
    )
    model.load_state_dict(
        torch.load(f"./models/weights/{model_filename}", map_location=device)
    )
    model.eval()
    transforms = get_transforms(
        options.size,
        mode=options.mode,
        intensity=options.intensity,
        mean=[0.2686, 0.2686, 0.2686],
        std=[0.1803, 0.1803, 0.1803],
    )
    dataset = MURADataset(
        dataframe=dataframe,
        data_path=os.path.join(options.data_path, f"AP/{options.data_subfolder}"),
        size=options.size,
        heatmap=options.heatmap,
        coefficient=options.coefficient,
        transforms=transforms,
        point_count=options.point_count,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=options.batch_size, shuffle=False,
    )
    return model, dataloader


def infer(dataloader, model, mode="max", verbose=True):
    """
    mode: how to obtain the landmark from heatmaps
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = None
    all_targets = None
    with torch.no_grad():
        for images, _, points in tqdm(dataloader):
            preds = model(images.to(device)).cpu()
            if all_preds is None:
                all_preds = preds
                all_targets = points
            else:
                all_preds = torch.cat([all_preds, preds], dim=0)
                all_targets = torch.cat([all_targets, points], dim=0)

    torch.save(all_preds, f"./dataset/heatmaps/heatmaps_timestamp_{time.time()}.pt") # saves all the heatmaps
    pred_coords = get_coords(all_preds, mode=mode)
    if verbose:
        alpha_error = calculate_alpha_error(pred_coords, all_targets)
        print(f"Error in angle (degrees): {alpha_error:.3f}")
        if pred_coords.size(1) == 4:
            ratio_error = calculate_ratio_error(pred_coords, all_targets)
            print(f"Error in ratio: {ratio_error:.5f}")

    return pred_coords, all_targets


def infer_one_fold(dataframe, model_filename, options):
    model, dataloader = build_model_dataloader(
        dataframe=dataframe, options=options, model_filename=model_filename
    )
    pred_coords, target_coords = infer(
        dataloader=dataloader, model=model, mode=options.infer_mode
    )
    return pred_coords, target_coords


def main(parser):
    options = parser.parse_args()

    for i, arg in enumerate(vars(options)):
        print(f"{i + 1}- {arg}, {getattr(options, arg)}")

    dataframe = pd.read_csv(f"./dataset/{options.df}")
    folds = options.folds
    if folds == 0:
        model_filename = options.model_filename
        pred_coords, target_coords = infer_one_fold(dataframe, model_filename, options)
    elif folds > 0:
        model_filenames = os.listdir(f"./models/weights/{options.model_filename}")
        all_pred_coords = []
        for name in model_filenames:
            pred_coords, target_coords = infer_one_fold(
                dataframe, f"{options.model_filename}/{name}", options
            )
            all_pred_coords.append(pred_coords)

        # there could be other strategies for aggregating the fold predictions
        # here I'll take the average of predictions after I gain the coordinates
        all_pred_coords = torch.stack(all_pred_coords, dim=0)
        pred_coords = all_pred_coords.mean(dim=0)
        alpha_error = calculate_alpha_error(pred_coords, target_coords)
        print(
            f"Final error in angle after aggregating the predictions (degrees): {alpha_error:.3f}"
        )
        if options.point_count == 4:
            ratio_error = calculate_ratio_error(pred_coords, target_coords)
            print(
                f"Final error in ratio after aggregating the predictions: {ratio_error:.5f}"
            )

    N, point_count = pred_coords.shape[:2]
    pred_coords = pred_coords.view(N, point_count * 2).tolist()
    target_coords = target_coords.view(N, point_count * 2).tolist()

    filenames = dataframe["#filename"].unique()
    inference_df = pd.DataFrame(
        {
            "#filename": filenames,
            f"pred_coords": pred_coords,
            f"target_coords": target_coords,
        }
    )
    inference_df.to_csv(
        f"./dataset/infer_{options.output}_model_{options.model}_size_{options.size}.csv",
        index=False,
    )


parser = inference_options()
if __name__ == "__main__":
    main(parser=parser)
