import torch
from torch import nn


class OutputNode(nn.Sequential):

    def __init__(
            self,
            in_channels: int,
            preconv_channels: int,
            conv_channels: int,
    ):
        super().__init__(
            nn.Conv2d(in_channels, preconv_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=preconv_channels),
            nn.Conv2d(preconv_channels, conv_channels, kernel_size=1),
        )


class CTDet(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, preconv_channels: int = 64):
        super().__init__()
        self.heatmap_node = OutputNode(in_channels, preconv_channels, num_classes)
        self.keypoint_node = OutputNode(in_channels, preconv_channels, 4)

    def forward(self, x: torch.Tensor):
        heatmap = self.heatmap_node(x)
        keypoint = self.keypoint_node(x)
        return {"heatmap": heatmap, "keypoint": keypoint}
