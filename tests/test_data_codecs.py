import numpy as np
import pytest
import torch

from gluon.codec.data import (CollateInputCodec, EmbeddingDataCodec,
                              HeatmapDataCodec)
from gluon.interfaces.data import GluonInputs, Location, TensorShape
from tests.utils import assert_vector_of_embeddings_equality_unordered


def test_location_object_stores_coordinates_correctly():

    BATCH_SIZE = 4
    NUM_CLASSES = 6
    SHAPE_X = 24
    SHAPE_Y = 32
    NUM_PEAKS = 12

    tensor_shape = BATCH_SIZE, NUM_CLASSES, SHAPE_X, SHAPE_Y

    peak_indices = torch.from_numpy(
        np.random.randint(
            low=[0, 0, 0, 0],
            high=tensor_shape,
            size=[NUM_PEAKS, 4],
        )
    )
    location = Location(
        center_xy=peak_indices[:, 2:4].float()
        / torch.tensor([SHAPE_X, SHAPE_Y], dtype=torch.float32),
        class_ids=peak_indices[:, 1],
        batch_idx=peak_indices[:, 0],
    )

    for i, (batch_idx, class_idx, x, y) in enumerate(peak_indices):
        assert location.center_xy[i, 0].item() == x / SHAPE_X
        assert location.center_xy[i, 1].item() == y / SHAPE_Y
        assert location.class_ids[i] == class_idx
        assert location.batch_idx[i] == batch_idx


def test_location_object_tensor_index_interface_works_correctly():

    BATCH_SIZE = 4
    NUM_CLASSES = 6
    SHAPE_X = 24
    SHAPE_Y = 32
    NUM_PEAKS = 12

    tensor_shape = BATCH_SIZE, NUM_CLASSES, SHAPE_X, SHAPE_Y

    peak_indices = torch.from_numpy(
        np.random.randint(
            low=[0, 0, 0, 0],
            high=tensor_shape,
            size=[NUM_PEAKS, 4],
        )
    )
    location = Location(
        center_xy=peak_indices[:, 2:4].float()
        / torch.tensor([SHAPE_X, SHAPE_Y], dtype=torch.float32),
        class_ids=peak_indices[:, 1],
        batch_idx=peak_indices[:, 0],
    )
    tensor_idx = location.as_tensor_indices(tensor_shape=TensorShape(*tensor_shape))
    assert torch.all(torch.eq(peak_indices, tensor_idx))


def test_heatmap_codec():
    BATCH_SIZE = 4
    NUM_CLASSES = 6
    SHAPE_X = 32
    SHAPE_Y = 64
    NUM_PEAKS = 12

    tensor_shape = BATCH_SIZE, NUM_CLASSES, SHAPE_X, SHAPE_Y

    peak_indices = torch.from_numpy(
        np.random.randint(
            low=[0, 0, 0, 0],
            high=tensor_shape,
            size=[NUM_PEAKS, 4],
        )
    )
    location = Location(
        center_xy=peak_indices[:, 2:4] / torch.from_numpy(np.array([SHAPE_X, SHAPE_Y])),
        class_ids=peak_indices[:, 1],
        batch_idx=peak_indices[:, 0],
    )

    codec = HeatmapDataCodec()
    encoded_heatmap = codec.encode(location, TensorShape(*tensor_shape))
    assert encoded_heatmap.size() == (BATCH_SIZE, NUM_CLASSES, SHAPE_X, SHAPE_Y)
    for batch_idx, class_idx, x, y in peak_indices:
        assert encoded_heatmap[batch_idx, class_idx, x, y] == 1

    result = codec.decode(encoded_heatmap, threshold=0.1)

    assert len(result.location.center_xy) == NUM_PEAKS

    assert_vector_of_embeddings_equality_unordered(
        result.location.center_xy,
        location.center_xy,
    )

    for class_idx in torch.unique(result.location.class_ids):
        num_decoded = sum(torch.eq(result.location.class_ids, class_idx))
        num_original = sum(torch.eq(location.class_ids, class_idx))
        assert num_decoded == num_original
    for batch_idx in torch.unique(result.location.batch_idx):
        num_decoded = sum(torch.eq(result.location.batch_idx, batch_idx))
        num_original = sum(torch.eq(location.batch_idx, batch_idx))
        assert num_decoded == num_original


def test_embedding_codec():
    BATCH_SIZE = 4
    NUM_CLASSES = 6
    SHAPE_X = 24
    SHAPE_Y = 32
    EMBEDDING_DIM = 3
    NUM_PEAKS = 12

    tensor_shape = BATCH_SIZE, NUM_CLASSES, SHAPE_X, SHAPE_Y

    peak_indices = torch.from_numpy(
        np.random.randint(
            low=[0, 0, 0, 0],
            high=tensor_shape,
            size=[NUM_PEAKS, 4],
        )
    )
    location = Location(
        center_xy=peak_indices[:, 2:4].float()
        / torch.tensor([SHAPE_X, SHAPE_Y], dtype=torch.float32),
        class_ids=peak_indices[:, 1],
        batch_idx=peak_indices[:, 0],
    )
    embeddings = torch.randn([NUM_PEAKS, EMBEDDING_DIM])

    codec = EmbeddingDataCodec()

    encoded_embeddings = codec.encode(
        location=location,
        embedding=embeddings,
        tensor_shape=TensorShape(*tensor_shape),
    )

    assert encoded_embeddings.size() == (BATCH_SIZE, EMBEDDING_DIM, SHAPE_X, SHAPE_Y)

    for i, (batch_idx, _, x, y) in enumerate(peak_indices):
        torch.testing.assert_allclose(
            encoded_embeddings[batch_idx, :, x, y],
            embeddings[i],
        )

    decoded_embeddings = codec.decode(encoded_embeddings, location)

    assert_vector_of_embeddings_equality_unordered(
        decoded_embeddings,
        embeddings,
    )


def test_image_input_collate_codec():
    BATCH_SIZE = 1
    CHANNEL_X = 2
    CHANNEL_Y = 3
    SPATIAL_X = 4
    SPATIAL_Y = 5
    images = [
        GluonInputs(
            elements={
                "x": torch.ones((CHANNEL_X, SPATIAL_X, SPATIAL_X)),
                "y": torch.zeros((CHANNEL_Y, SPATIAL_Y, SPATIAL_Y)),
            }
        )
        for _ in range(BATCH_SIZE)
    ]

    codec = CollateInputCodec()

    inputs = codec.encode(images)

    assert isinstance(inputs, dict)
    assert len(inputs) == 2
    assert "x" in inputs and "y" in inputs
    assert inputs["x"].size() == (BATCH_SIZE, CHANNEL_X, SPATIAL_X, SPATIAL_X)
    assert inputs["y"].size() == (BATCH_SIZE, CHANNEL_Y, SPATIAL_Y, SPATIAL_Y)
    assert torch.all(inputs["x"] == 1)
    assert torch.all(inputs["y"] == 0)


if __name__ == "__main__":
    pytest.main()
