from typing import List

import numpy as np
from torchvision.transforms import functional as tvf
from torchvision.transforms import InterpolationMode

from .interfaces.data import GluonDataPoint
from .interfaces.transformation import Transformation


class ResizeForTraining(Transformation):

    def __init__(
        self,
        input_element_key: str,
        target_size_wh: List[int],
        resize_mode: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.target_size_hw = target_size_wh[::-1]
        self.resize_mode = resize_mode
        self.key = input_element_key

    def forward(self, datapoint: GluonDataPoint) -> GluonDataPoint:
        tensor = datapoint.inputs.elements[self.key]
        depth, h, w = tensor.size()
        if h > w:
            target_h = self.target_size_hw[0]
            target_w = int(round((w/h) * self.target_size_hw[1]))
            pad_h = 0
            pad_w = self.target_size_hw[1] - target_w
        else:
            target_h = int(round((h/w) * self.target_size_hw[0]))
            target_w = self.target_size_hw[1]
            pad_h = self.target_size_hw[0] - target_h
            pad_w = 0
        new_tensor = tvf.resize(tensor, (target_h, target_w), self.resize_mode)
        new_padded_tensor = tvf.pad(new_tensor, padding=[0, 0, pad_w, pad_h])
        padded_tensor_shape = new_padded_tensor.size()
        assert padded_tensor_shape == (depth, self.target_size_hw[0], self.target_size_hw[1]), f"{padded_tensor_shape}"
        datapoint.inputs.elements[self.key] = new_padded_tensor

        box_corners_scaler = np.array(
            [target_h / (target_h + pad_h), target_w / (target_w + pad_w)] * 2  # [4] Matrix form
        )
        assert np.all(box_corners_scaler <= 1)
        datapoint.labels.box_corners *= box_corners_scaler[None, :]
        return datapoint

    def reverse(self, data_point: GluonDataPoint) -> GluonDataPoint:
        pass
