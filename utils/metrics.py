import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error

from .measure import get_angle, get_ratio

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_landmark(pred):
    pred = pred.squeeze(0)
    indices = torch.nonzero(pred == pred.max())[0]
    return indices

def get_coords(preds, mode='max'):
    if mode == 'max':
        indices_pred = []
        for pred in preds:
            indices_pred_ = [get_landmark(channel) for channel in pred]
            indices_pred.append(torch.stack(indices_pred_, dim=0))

        counts = preds.size(0)
        indices_pred = torch.stack(indices_pred, dim=0).view(counts, -1, 2).float()
        indices_pred = indices_pred.flip(dims=[-1])
        return indices_pred

def get_metrics(preds, points):
    indices_pred = get_coords(preds, mode='max')

    mae = F.l1_loss(indices_pred, points)
    
    alpha_target = [get_angle(points_[0], points_[1], points_[2]) for points_ in points]
    alpha_pred = [get_angle(points_[0], points_[1], points_[2]) for points_ in indices_pred]
    alpha_error = mean_absolute_error(alpha_pred, alpha_target)
    
    return mae, alpha_error

def calculate_alpha_error(pred_coords, target_coords):
    alpha_target = [
        get_angle(points_[0], points_[1], points_[2]) for points_ in target_coords
    ]
    alpha_pred = [
        get_angle(points_[0], points_[1], points_[2]) for points_ in pred_coords
    ]
    alpha_error = mean_absolute_error(alpha_pred, alpha_target)
    return alpha_error

def calculate_ratio_error(pred_coords, target_coords):
    ratio_target = [get_ratio(points)['AI_ratio'] for points in target_coords]
    ratio_pred = [get_ratio(points)['AI_ratio'] for points in pred_coords]
    ratio_error = mean_absolute_error(ratio_pred, ratio_target)
    return ratio_error