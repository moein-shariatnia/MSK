import os
from ast import literal_eval
import numpy as np

import torch
from torch.utils.data import Dataset

from .data_utils import read_image, build_gaussian_heatmap, build_distance_heatmap


class MURADataset(Dataset):
    def __init__(
        self,
        dataframe,
        data_path,
        size,
        heatmap="gaussian",
        coefficient=None,
        transforms=None,
        point_count=3,
    ):
        self.data_path = data_path
        self.dataframe = dataframe
        self.point_count = point_count
        self.file_names = dataframe["#filename"].unique()
        self.points = [self.get_points(filename) for filename in self.file_names]
        self.transforms = transforms
        self.heatmap = heatmap
        self.size = size
        self.coefficient = coefficient
        

    def __getitem__(self, idx):
        filename = os.path.join(self.data_path, self.file_names[idx])
        points = self.points[idx]
        image = read_image(filename)
        if self.transforms is not None:
            transformed = self.transforms(image=image, keypoints=points)
            image = transformed["image"]
            points = transformed["keypoints"]

        points = np.round(
            points, decimals=2
        )  # To avoid numerical issues in gaussian heatmap function

        image = torch.tensor(image).float().permute(2, 0, 1)
        heatmaps = []

        if self.heatmap == "gaussian":
            for point in points:
                heatmap = build_gaussian_heatmap(
                    self.size, point, sigma=self.coefficient
                )
                heatmaps.append(torch.tensor(heatmap))

        elif self.heatmap == "distance":
            for point in points:
                heatmap = build_distance_heatmap(
                    self.size, point, gamma=self.coefficient
                )
                heatmaps.append(torch.tensor(heatmap))

        heatmap = torch.stack(heatmaps, dim=0).float()
        points = torch.tensor(points).float()
        return image, heatmap, points

    def __len__(self):
        return len(self.file_names)

    def get_points(self, filename):
        sample_df = self.dataframe[self.dataframe["#filename"] == filename]
        sample_df = sample_df.sort_values(by="region_id", ascending=True)
        points = sample_df["region_shape_attributes"].tolist()
        points = [literal_eval(point) for point in points]
        points = [(point["cx"], point["cy"]) for point in points[: self.point_count]]
        return points
