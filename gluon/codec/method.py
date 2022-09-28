"""
Model-specific codecs.
Meant to be used with a specific network head.
"""
from typing import Dict, List

import numpy as np
import torch

from ..interfaces.data import GluonInputs, GluonLabels, Location, TensorShape
from ..operation import (generate_relative_corner_position,
                         restore_corner_absolute_position,
                         snap_coordinates_to_grid)
from .data import CollateInputCodec, EmbeddingDataCodec, HeatmapDataCodec

__all__ = ["CTDetCodec", "ModelCodec"]


class ModelCodec:
    def encode_inputs(self, inputs: List[GluonInputs]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def decode_inputs(self, tensor: Dict[str, torch.Tensor]) -> List[GluonInputs]:
        raise NotImplementedError

    def encode_labels(self, labels: List[GluonLabels]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def decode_labels(self, raw_prediction: Dict[str, torch.Tensor]) -> List[GluonLabels]:
        raise NotImplementedError


class CTDetCodec(ModelCodec):
    def __init__(
        self,
        image_shape: TensorShape,
        label_tensor_shape: TensorShape,
        num_classes: int,
    ):
        self.image_shape = image_shape
        self.tensor_shape = label_tensor_shape
        self.tensor_shape_scaler = label_tensor_shape.width, label_tensor_shape.height
        self.num_classes = num_classes
        self.image_codec = CollateInputCodec()
        self.heatmap_codec = HeatmapDataCodec()
        self.embedding_codec = EmbeddingDataCodec()

    def encode_inputs(self, inputs: List[GluonInputs]) -> Dict[str, torch.Tensor]:
        return self.image_codec.encode(inputs)

    def decode_inputs(
        self, encoded_inputs: Dict[str, torch.Tensor]
    ) -> List[GluonInputs]:
        return self.image_codec.decode(encoded_inputs)

    def encode_labels(self, data: List[GluonLabels]):
        batch_size = len(data)
        centers = []
        corner_offsets = []
        class_ids = []
        batch_ids = []
        for batch_id, d in enumerate(data):
            corners = d.box_corners

            assert len(d.box_corners) > 0, f"Box corners len: {len(d.box_corners)}"
            centers.append((corners[:, :2] + corners[:, 2:]) / 2.0)
            corner_offsets.append(
                generate_relative_corner_position(
                    corners,
                    snap_coordinates_to_grid(
                        centers[-1],
                        grid_resolution=self.tensor_shape_scaler,
                    ),
                )
            )
            batch_ids.append([batch_id] * len(corners))
            class_ids.append(d.class_ids)

        locations = Location(
            center_xy=torch.from_numpy(
                np.concatenate(centers, axis=0).astype(np.float32)
            ),
            class_ids=torch.as_tensor(np.concatenate(class_ids, axis=0).astype(int)),
            batch_idx=torch.as_tensor(np.concatenate(batch_ids, axis=0).astype(int)),
        )

        heatmap_tensor_shape = TensorShape(
            batch_size,
            self.num_classes,
            self.tensor_shape.width,
            self.tensor_shape.height,
        )
        keypoint_tensor_shape = TensorShape(
            batch_size,
            4,
            self.tensor_shape.width,
            self.tensor_shape.height,
        )
        result = dict(
            heatmap=self.heatmap_codec.encode(
                location=locations, tensor_shape=heatmap_tensor_shape
            ),
            keypoint=self.embedding_codec.encode(
                location=locations,
                embedding=torch.from_numpy(np.concatenate(corner_offsets, axis=0)),
                tensor_shape=keypoint_tensor_shape,
            ),
        )

        return result

    def decode_labels(
        self, raw_prediction: Dict[str, torch.Tensor]
    ) -> List[GluonLabels]:
        peak_find_result = self.heatmap_codec.decode(
            heatmap=raw_prediction["heatmap"],
            threshold=0.1,
        )
        box_corners = self.embedding_codec.decode(
            embedding_tensor=raw_prediction["keypoint"],
            location=peak_find_result.location,
        ).numpy()

        per_batch_results = []
        for batch_idx in range(len(raw_prediction["keypoint"])):
            batch_mask = peak_find_result.location.batch_idx == batch_idx
            batch_centers = peak_find_result.location.center_xy[batch_mask].numpy()
            batch_class_ids = peak_find_result.location.class_ids[batch_mask].numpy()
            batch_scores = peak_find_result.scores[batch_mask].numpy()
            batch_corners = box_corners[batch_mask]
            corners = restore_corner_absolute_position(batch_corners, batch_centers)
            gl_label = GluonLabels(
                box_corners=corners,
                class_ids=batch_class_ids,
                scores=batch_scores,
            )
            per_batch_results.append(gl_label)

        return per_batch_results
