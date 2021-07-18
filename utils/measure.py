import cv2
import math
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2, degrees=True):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle_radian = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if not degrees:
        return angle_radian
    angle_degrees = math.degrees(angle_radian)
    if np.isnan(angle_degrees):
        return 180
    return angle_degrees

def get_angle(center, point1, point2):
    x0, y0 = center
    x1, y1 = point1
    x2, y2 = point2

    v1 = np.array([x1 - x0, y1 - y0])
    v2 = np.array([x2 - x0, y2 - y0])

    return angle_between(v1, v2)

def get_line_formula(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    slope = (y2 - y1) / (x2 - x1)
    bias = y2 - slope * x2
    return slope, bias

def get_ratio(points, verbose=False):
    slope01, bias01 = get_line_formula(points[0], points[1])
    is_inf = False
    if slope01 == -float("inf") or slope01 == float("inf"):
        if verbose:
            print("Inf Slope!")
        is_inf = True
        slope01 = 999 * np.sign(slope01)
    
    bias3_ = points[3][1] - slope01 * points[3][0]
    y3_ = points[0][1] #
    x3_ = (y3_ - bias3_) / slope01

    perp_line_slope = -1 / slope01
    perp_line_bias0 = points[0][1] - perp_line_slope * points[0][0]

    y01 = points[0][1] #
    x01 = (y01 - bias01) / slope01

    x_share3_ = (bias3_ - perp_line_bias0) / (perp_line_slope - slope01)
    y_share3_ = slope01 * x_share3_ + bias3_

    bias2_ = points[2][1] - slope01 * points[2][0]
    y2_ = points[0][1] #
    x2_ = (y2_ - bias2_) / slope01

    perp_line_bias1 = points[1][1] - perp_line_slope * points[1][0]

    x_share2_ = (bias2_ - perp_line_bias1) / (perp_line_slope - slope01)
    y_share2_ = slope01 * x_share2_ + bias2_

    GH_vector = [x_share3_ - points[0][0], y_share3_ - points[0][1]]
    GA_vector = [x_share2_ - points[1][0], y_share2_ - points[1][1]]

    GH_length = np.linalg.norm(GH_vector)
    GA_length = np.linalg.norm(GA_vector)
    AI_ratio = GA_length / GH_length
    output = {
        'x3_': x3_,
        'y3_': y3_,
        'x01': x01,
        'y01': y01,
        'x_share3_': x_share3_,
        'y_share3_': y_share3_,
        'x2_': x2_,
        'y2_': y2_,
        'x_share2_': x_share2_,
        'y_share2_': y_share2_,
        'is_inf': is_inf,
        'AI_ratio': AI_ratio
    }
    return output