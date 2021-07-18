import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, model, pretrained, classes=3, sigmoid_coeff=1.0):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=model, encoder_weights=pretrained, in_channels=3, classes=classes,
        )  # check how I can train from scratch
        self.sigmoid_coeff = sigmoid_coeff

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x) * self.sigmoid_coeff