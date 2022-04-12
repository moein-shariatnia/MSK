import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from .measure import get_angle, get_ratio


def show_angle_ratio(
    image_path,
    size,
    points,
    ax=None,
    lines=True,
    angle=False,
    verbose=True,
    return_values=False,
):
    """
    image_path: complete path to the image
    size: must be mentioned for resizing
    points: as array or tensor with shape (4, 2)
    verbose: whether to print out angle and ratio
    """
    image = cv2.imread(image_path)[..., ::-1]
    transform = A.Resize(size, size, always_apply=True)
    image = transform(image=image)["image"]
    alpha = get_angle(points[0], points[1], points[2])
    output = get_ratio(points)
    if verbose:
        print(
            f"Angle (degrees): {alpha:.3f}\n"
            f"AI Ratio: {output['AI_ratio']:.3f}\n"
            f"Is Inf: {output['is_inf']}"
        )

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(image, cmap="gray")
    ax.scatter(points[:, 0], points[:, 1], c="g")  # g for main points
    ax.plot(points[[0, 1], 0], points[[0, 1], 1], c="k")
    if lines:
        ax.plot([points[3][0], output["x3_"]], [points[3][1], output["y3_"]], c="k")
        ax.plot([points[0][0], output["x01"]], [points[0][1], output["y01"]], c="k")

        ax.scatter(
            output["x_share3_"], output["y_share3_"], c="r"
        )  # r for rule-obtained points
        ax.plot(
            [points[0][0], output["x_share3_"]],
            [points[0][1], output["y_share3_"]],
            c="k",
        )

        ax.plot([points[2][0], output["x2_"]], [points[2][1], output["y2_"]], c="k")

        ax.plot(
            [points[1][0], output["x_share2_"]],
            [points[1][1], output["y_share2_"]],
            c="k",
        )
        ax.scatter(output["x_share2_"], output["y_share2_"], c="r")
    if angle:
        ax.plot(points[[0, 2], 0], points[[0, 2], 1], c="k")

    if return_values:
        return alpha, output["AI_ratio"], output["is_inf"]


def show_heatmap_points(
    image_path,
    size,
    image_heatmaps,
    pred_points,
    target_points=None,
    ax=None,
    alpha=0.2,
):
    """
    image_path: complete path to the image
    image_heatmaps: heatmaps for the image with the shape (4, size, size)
    pred_points: predicted points with shape (4, 2)
    target_points: target points with shape (4, 2)
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    image = cv2.imread(image_path)[..., ::-1]
    transform = A.Resize(size, size, always_apply=True)
    image = transform(image=image)["image"]

    ax.imshow(image)
    ax.matshow(image_heatmaps.sum(0), alpha=alpha, cmap="gray")  # could be other things
    ax.scatter(pred_points[:, 0], pred_points[:, 1], s=2, c="r", label="Prediction")
    if target_points is not None:
        ax.scatter(target_points[:, 0], target_points[:, 1], s=2, c="b", label="Target")
    ax.legend()
