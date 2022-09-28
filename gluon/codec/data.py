"""
Non-model specific codecs, used to build more complex, model-specific codecs.
These are meant to be reusable.
"""

import abc
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from torch.nn import functional as F

from gluon import operation
from gluon.interfaces import data
from gluon.interfaces.data import Location, TensorShape


class DataCodec(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        pass


class LambdaInputCodec(DataCodec):
    def __init__(
        self,
        encode_func: Callable[[List[data.GluonInputs]], Dict[str, torch.Tensor]],
        decode_func: Optional[
            Callable[[Dict[str, torch.Tensor]], List[data.GluonInputs]]
        ] = None,
    ):
        self.encode_func = encode_func
        self.decode_func = decode_func

    def encode(self, inputs: List[data.GluonInputs]) -> Dict[str, torch.Tensor]:
        return self.encode_func(inputs)

    def decode(self, encoded_inputs: Dict[str, torch.Tensor]) -> List[data.GluonInputs]:
        if self.decode_func is None:
            raise RuntimeError("Decode function unspecified.")
        return self.decode_func(encoded_inputs)


class CollateInputCodec(DataCodec):
    def encode(self, inputs: List[data.GluonInputs]) -> Dict[str, torch.Tensor]:
        collection = defaultdict(list)
        for input_object in inputs:
            for key, value in input_object.elements.items():
                assert isinstance(value, torch.Tensor)
                assert len(value.size()) == 3
                collection[key].append(value)
        result = {
            key: torch.stack(tensor_list, dim=0)
            for key, tensor_list in collection.items()
        }
        return result

    def decode(self, encoded_inputs: Dict[str, torch.Tensor]):
        # TODO: implement simple decollate
        raise NotImplementedError


class HeatmapDataCodec(DataCodec):
    class Result(NamedTuple):
        location: Location
        scores: torch.Tensor

    def encode(
        self,
        location: Location,
        tensor_shape: data.TensorShape,
    ) -> torch.Tensor:
        """
        :param location: Location
            Describes where to put the peak centers (the gauss maxima)
        :param tensor_shape: Tuple[int x4]
            batch_size, num_classes, spatial_x, spatial_y
        :return: Tensor
            Encoded heatmaps
        """
        N = len(location.center_xy)
        heatmap = operation.gauss_2D(
            location.as_tensor_indices(tensor_shape),
            sigma=torch.ones(N),
            tensor_shape=tensor_shape,
            min_threshold=0.1,
        )
        return heatmap

    def decode(self, heatmap: torch.Tensor, threshold: float) -> Result:
        tensor_size = heatmap.size()  # [batch_size, num_classes, spatial_x, spatial_y]
        scaling_values = torch.as_tensor(tensor_size[2:4], dtype=torch.float32)
        max_filtered: torch.Tensor = F.max_pool2d(
            heatmap,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        local_maxima = torch.eq(max_filtered, heatmap)
        score_over_threshold = torch.gt(heatmap, threshold)
        dense_peak_locations = torch.logical_and(local_maxima, score_over_threshold)
        peak_locations = dense_peak_locations.argwhere()
        scores = torch.as_tensor([heatmap[b, c, x, y] for b, c, x, y in peak_locations])
        batch_idx = peak_locations[:, 0]
        class_ids = peak_locations[:, 1]
        center_xy = peak_locations[:, 2:4] / scaling_values[None, :]
        # center_xy = torch.flip(center_yx, dims=[1])
        result = self.Result(
            Location(
                center_xy=center_xy,
                class_ids=class_ids,
                batch_idx=batch_idx,
            ),
            scores,
        )
        return result


class EmbeddingDataCodec(DataCodec):
    def encode(
        self,
        location: Location,
        embedding: torch.Tensor,
        tensor_shape: data.TensorShape,
    ) -> torch.Tensor:
        result = operation.place_embeddings(
            center_locations=location.as_tensor_indices(tensor_shape),
            embeddings=embedding,
            shape=tensor_shape,
        )
        return result

    def decode(
        self,
        embedding_tensor: torch.Tensor,
        location: Location,
    ) -> torch.Tensor:
        tensor_shape = TensorShape.from_tensor(
            embedding_tensor
        )  # [batch_size, embedding_depth, spatial_x, spatial_y]
        tensor_indices = location.as_tensor_indices(tensor_shape=tensor_shape)
        embeddings = torch.stack(
            [embedding_tensor[b, :, x, y] for b, _, x, y in tensor_indices], dim=0
        )  # [n, d]
        assert embeddings.size(1) == embedding_tensor.size(1)
        return embeddings
