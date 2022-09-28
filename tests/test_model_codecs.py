import numpy as np
import pytest
import torch

from gluon.codec.method import CTDetCodec
from gluon.interfaces.data import GluonInputs, GluonLabels, TensorShape
from gluon.operation import snap_coordinates_to_grid
from tests import utils as testing_utils


def test_ctdet_codec_input_encoding():
    IMAGE_CHANNEL = 3
    IMAGE_SPATIAL_X = 32
    IMAGE_SPATIAL_Y = 24

    LABEL_SPATIAL_X = 8
    LABEL_SPATIAL_Y = 6
    NUM_CLASSES = 1

    BATCH_SIZE = 3

    inputs = []
    for i in range(BATCH_SIZE):
        image = torch.tensor(
            np.random.random((IMAGE_SPATIAL_X, IMAGE_SPATIAL_Y, IMAGE_CHANNEL)),
            dtype=torch.float32,
        )
        inputs.append(GluonInputs(elements={"x": image}))

    codec = CTDetCodec(
        image_shape=TensorShape(None, IMAGE_CHANNEL, IMAGE_SPATIAL_X, IMAGE_SPATIAL_Y),
        label_tensor_shape=TensorShape(None, None, LABEL_SPATIAL_X, LABEL_SPATIAL_Y),
        num_classes=NUM_CLASSES,
    )

    encoded_images = codec.encode_inputs(inputs)
    assert all(
        torch.all(encoded_images["x"][i] == inputs[i].elements["x"])
        for i in range(BATCH_SIZE)
    )


def test_ctdet_codec_label():
    IMAGE_CHANNEL = 3
    IMAGE_SPATIAL_X = 64
    IMAGE_SPATIAL_Y = 48

    LABEL_SPATIAL_X = 16
    LABEL_SPATIAL_Y = 12

    NUM_CLASSES = 4

    BATCH_SIZE = 5

    PER_BATCH_CORNERS = [
        [(1, 15, 11, 15, 11), (2, 0, 0, 0, 0)],
        [],
        [(0, 6, 0, 10, 4), (3, 0, 6, 4, 10)],
        [],
        [(0, 12, 8, 15, 11)],
    ]

    label_shape = np.array([LABEL_SPATIAL_X, LABEL_SPATIAL_Y]).astype(np.float32)

    labels = []
    for i in range(BATCH_SIZE):

        boxes = PER_BATCH_CORNERS[i]
        if boxes:
            boxes_np = np.array(boxes)
        else:
            boxes_np = np.zeros([0, 5], dtype=int)
        box_x0y0 = boxes_np[:, 1:3].astype(np.float32) / label_shape
        box_x1y1 = boxes_np[:, 3:5].astype(np.float32) / label_shape
        box_corners = np.concatenate([box_x0y0, box_x1y1], axis=1)
        class_ids = boxes_np[:, 0]
        labels.append(
            GluonLabels(box_corners=box_corners, class_ids=class_ids, scores=None)
        )

    codec = CTDetCodec(
        image_shape=TensorShape(None, IMAGE_CHANNEL, IMAGE_SPATIAL_X, IMAGE_SPATIAL_Y),
        label_tensor_shape=TensorShape(None, None, LABEL_SPATIAL_X, LABEL_SPATIAL_Y),
        num_classes=NUM_CLASSES,
    )

    encoded_labels = codec.encode_labels(labels)

    encoded_heatmap = encoded_labels["heatmap"]
    encoded_embeddings = encoded_labels["keypoints"]

    assert encoded_heatmap.size() == (
        BATCH_SIZE,
        NUM_CLASSES,
        LABEL_SPATIAL_X,
        LABEL_SPATIAL_Y,
    )
    for batch_idx, label in enumerate(labels):
        for class_id, corners in zip(label.class_ids, label.box_corners):
            x = int(round(sum(corners[0::2]) * LABEL_SPATIAL_X / 2.0))
            y = int(round(sum(corners[1::2]) * LABEL_SPATIAL_Y / 2.0))
            assert encoded_heatmap[batch_idx, class_id, x, y] == 1

    assert encoded_embeddings.size() == (
        BATCH_SIZE,
        4,
        LABEL_SPATIAL_X,
        LABEL_SPATIAL_Y,
    )
    for batch_idx, label in enumerate(labels):
        for class_id, corners in zip(label.class_ids, label.box_corners):
            center_low = snap_coordinates_to_grid(
                ((corners[:2] + corners[2:4]) / 2.0)[None, :],
                grid_resolution=(LABEL_SPATIAL_X, LABEL_SPATIAL_Y),
            )[0]

            center_x_high = int(round(center_low[0] * LABEL_SPATIAL_X))
            center_y_high = int(round(center_low[1] * LABEL_SPATIAL_Y))
            embedded_corners = encoded_embeddings[
                batch_idx, :, center_x_high, center_y_high
            ].numpy()
            expectation = np.concatenate(
                [
                    corners[:2] - center_low,
                    corners[2:4] - center_low,
                ],
                axis=0,
            )
            np.testing.assert_allclose(embedded_corners, expectation, atol=1e-5)

    decoded_labels = codec.decode_labels(encoded_labels)
    for iteration, (label, decoded_label) in enumerate(zip(labels, decoded_labels)):
        assert len(label) == len(decoded_label)
        np.testing.assert_allclose(
            decoded_label.scores, np.ones_like(decoded_label.scores)
        )
        testing_utils.assert_vector_of_embeddings_equality_unordered(
            torch.tensor(label.box_corners),
            torch.tensor(decoded_label.box_corners),
        )


if __name__ == "__main__":
    pytest.main()
