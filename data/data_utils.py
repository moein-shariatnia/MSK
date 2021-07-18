import cv2
import numpy as np


def read_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def build_gaussian_heatmap(size, point, sigma):
    heatmap = np.zeros((size, size))

    upper_left = [int(point[0] - 3 * sigma), int(point[1] - 3 * sigma)]
    bottom_right = [int(point[0] + 3 * sigma + 1), int(point[1] + 3 * sigma + 1)]

    if (
        upper_left[0] >= size
        or upper_left[1] >= size
        or bottom_right[0] < 0
        or bottom_right[1] < 0
    ):
        return heatmap

    gaussian_size = 6 * sigma + 1
    x = np.arange(0, gaussian_size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = gaussian_size // 2
    gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -upper_left[0]), min(bottom_right[0], size) - upper_left[0]
    g_y = max(0, -upper_left[1]), min(bottom_right[1], size) - upper_left[1]

    h_x = max(0, upper_left[0]), min(bottom_right[0], size)
    h_y = max(0, upper_left[1]), min(bottom_right[1], size)

    heatmap[h_y[0] : h_y[1], h_x[0] : h_x[1]] = gaussian[
        g_y[0] : g_y[1], g_x[0] : g_x[1]
    ]

    return heatmap


def build_distance_heatmap(size, point, gamma):
    x = np.linspace(1, size, size)
    y = np.linspace(1, size, size)
    xx, yy = np.meshgrid(x, y)
    xx = (xx - point[0]) ** 2
    yy = (yy - point[1]) ** 2
    array = np.sqrt(xx + yy)
    max_values = array.max()
    normalized = (1 - array / max_values) ** gamma
    return normalized