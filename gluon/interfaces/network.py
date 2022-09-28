from typing import Dict

import torch
from torch import nn

from .data import TensorShape


class ModelInterface(nn.Module):
    def __init__(
        self,
        input_shape: TensorShape,
        output_shape: TensorShape,
        name: str,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def report_metrics(self) -> Dict[str, torch.Tensor]:
        return {}
