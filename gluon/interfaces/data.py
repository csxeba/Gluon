import dataclasses
from typing import Any, Dict, NamedTuple, Optional, List

import numpy as np
import pydantic
import torch


class Config(pydantic.BaseConfig):
    arbitrary_types_allowed = True


@dataclasses.dataclass
class TensorShape:
    batch_size: Optional[int]
    depth: Optional[int]
    width: int
    height: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        shape = tensor.size()
        if len(shape) < 2:
            raise ValueError("Tensor must be at least 2 dimensional.")
        if len(shape) > 4:
            raise ValueError("Tensor must be at most 4 dimensional.")
        normalized_shape = ((None,) * (4 - len(shape))) + shape
        return cls(*normalized_shape)

    def as_tuple(self):
        return self.batch_size, self.depth, self.width, self.height


# noinspection PyMethodParameters
class Location(pydantic.BaseModel):
    """
    Location of a 'pixel' in a 4D tensor
    """

    Config = Config

    center_xy: torch.Tensor  # float32, scaled to 0..1
    class_ids: torch.Tensor  # int
    batch_idx: torch.Tensor  # int

    @pydantic.validator("center_xy")
    def _validate_center_xy(cls, v: torch.Tensor, values: Dict[str, Any]):

        if v.dtype != torch.float32:
            raise TypeError(f"center_xy must be a float32 tensor. Got: {v.dtype}")
        if v.size(1) != 2:
            raise ValueError("center_xy must be of shape [n, 2]")
        if torch.any(torch.logical_or(v < 0, v >= 1.0)):
            raise ValueError("center_xy must be within range (0 .. 1(")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(f"Length of center_xy must match length of {key}")
        return v

    @pydantic.validator("class_ids")
    def _validate_class_ids(cls, v: torch.Tensor, values: Dict[str, Any]):
        if v.dtype != torch.int64:
            raise TypeError(f"class_ids must be an int64 tensor. Got: {v.dtype}")
        if len(v.size()) != 1:
            raise ValueError("center_xy must be of shape [n]")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(f"Length of class_ids must match length of {key}")
        return v

    @pydantic.validator("batch_idx")
    def _validate_batch_idx(cls, v: torch.Tensor, values: Dict[str, Any]):
        if v.dtype != torch.int64:
            raise TypeError(f"batch_idx must be an int64 tensor. Got: {v.dtype}")
        if len(v.size()) != 1:
            raise ValueError("batch_idx must be of shape [n]")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(f"Length of batch_idx must match length of {key}")
        return v

    def as_tensor_indices(self, tensor_shape: TensorShape) -> torch.Tensor:
        shape_scaler = torch.as_tensor(
            tensor_shape.as_tuple()[2:4], dtype=torch.float32
        )
        center_xy_scaled = torch.clamp(
            torch.round(self.center_xy * shape_scaler[None, :]),
            torch.zeros(2), shape_scaler - 1,
        ).int()
        tensor_indices = torch.cat(
            [
                self.batch_idx[:, None],
                self.class_ids[:, None],
                center_xy_scaled,
            ],
            dim=1,
        )
        return tensor_indices


# noinspection PyMethodParameters
class GluonLabels(pydantic.BaseModel):
    """
    box_corners: NDArray[float], shape: [n, 4]
        Absolute box corner coordinates scaled to 0. - 1. by dividing with image shapes.
        Dimension 0 corresponds to all objects on an image
        Dimension 1 is [x0, y0, x1, y1]
    class_ids: NDArray[int], shape: [n]
        Unmapped class IDs.
        Dimension 0 corresponds to all objects on an image
    scores: NDArray[float], shape: [n], Optional
        All 1s for ground truth and object confidence for prediction.
        Dimension 0 corresponds to all objects on an image
        Can be set to None in case of ground truth.
    """

    Config = Config

    box_corners: np.ndarray
    class_ids: np.ndarray
    scores: Optional[np.ndarray]

    @pydantic.validator("box_corners")
    def _validate_box_corners(cls, v, values):
        if v.dtype != np.float32:
            raise TypeError(f"box_corners must be of type np.float32. Got: {v.dtype}")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(
                        f"Length of box_corners must match length of {key}"
                    )
        return v

    @pydantic.validator("class_ids")
    def _validate_class_ids(cls, v, values):
        if v.ndim != 1:
            raise ValueError(f"class_ids must be 1 dimensional. Received: {v[:3]}, ...")
        if v.dtype != np.int64:
            raise TypeError(f"class_ids must be of type int64. Got: {v.dtype}")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(f"Length of class_ids must match length of {key}")
        return v

    @pydantic.validator("scores")
    def _validate_scores(cls, v, values):
        if v is None:
            return v
        if v.dtype != np.float32:
            raise TypeError(f"scores must be of type np.float32. Got: {v.dtype}")
        if values:
            for key, value in values.items():
                if len(value) != len(v):
                    raise ValueError(f"Length of scores must match length of {key}")
        return v

    def __len__(self) -> int:
        return len(self.box_corners)


# noinspection PyMethodParameters
class GluonInputs(pydantic.BaseModel):

    Config = Config

    elements: Dict[str, torch.Tensor]

    @pydantic.validator("elements")
    def _validate_input_ranges(cls, v: Dict[str, torch.Tensor]):
        if not all(tensor.dtype == torch.float32 for tensor in v.values()):
            dictstr = "\n  ".join(
                f"{key}: {type(value.dtype)}" for key, value in v.items()
            )
            raise TypeError(f"Input has wrong dtype(s):\n{dictstr}")
        for key, value in v.items():
            tensor_range = torch.min(value), torch.max(value)
            if tensor_range[0] < 0.0 or tensor_range[1] > 1.0:
                raise ValueError(
                    f"Input tensor {key} has invalid range: min={tensor_range[0]} max={tensor_range[1]}"
                )
        return v


class GluonDataPoint(NamedTuple):
    metadata: dict
    inputs: GluonInputs
    labels: GluonLabels

    @classmethod
    def make_empty(cls, metadata: dict, input_element_keys: List[str]):
        gluon_input = GluonInputs(elements={
            key: torch.empty(dtype=torch.float32) for key in input_element_keys
        })
        gluon_label = GluonLabels(
            box_corners=np.empty((0, 4), dtype=np.float32),
            class_ids=np.empty(0, dtype=int),
            scores=None,
        )
        return GluonDataPoint(metadata=metadata, inputs=gluon_input, labels=gluon_label)
