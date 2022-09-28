import math

import numpy as np
import pytest
import torch

from gluon.interfaces.data import TensorShape
from gluon.operation import (gauss_1D, gauss_2D,
                             generate_relative_corner_position,
                             place_embeddings,
                             restore_corner_absolute_position)


def test_gauss_1D():

    SPATIAL_DIM = 10
    NUM_COORDS = 3
    center_idx = np.arange(0, NUM_COORDS).astype(int)
    center_idx = center_idx + SPATIAL_DIM // 2 - NUM_COORDS // 2

    gausses = gauss_1D(
        center_coords_1D=torch.from_numpy(center_idx),
        sigma=torch.full([len(center_idx)], 1.0),
        shape=SPATIAL_DIM,
        min_threshold=0.1,
    )

    assert gausses.size(0) == NUM_COORDS
    assert gausses.size(1) == SPATIAL_DIM
    for gauss, center_ind in zip(gausses, center_idx):
        assert math.isclose(gauss[center_ind], 1.0)
        patch = gauss[center_ind - 1 : center_ind + 2]
        assert torch.all(patch <= 1.0) and torch.all(patch > 0.0)
        print(patch)


def test_gauss_2D():
    BATCH_SIZE = 3
    CLASSES = 4
    SPATIAL = 13, 17
    NUM_OBJECTS = 4
    tensor_shape = (BATCH_SIZE, CLASSES) + SPATIAL
    obj_coords = np.random.randint(
        low=np.zeros(NUM_OBJECTS),
        high=tensor_shape,
        size=[NUM_OBJECTS, 4],
    )
    np.testing.assert_array_less(np.max(obj_coords, axis=0), tensor_shape)
    gausses = gauss_2D(
        center_locations=torch.from_numpy(obj_coords),
        sigma=torch.full([len(obj_coords)], 1.0),
        tensor_shape=TensorShape(*tensor_shape),
        min_threshold=0.1,
    )
    assert gausses.size(0) == BATCH_SIZE
    assert gausses.size(1) == CLASSES
    assert gausses.size(2) == SPATIAL[0]
    assert gausses.size(3) == SPATIAL[1]
    for batch_idx, class_idx, x, y in obj_coords:
        assert math.isclose(gausses[batch_idx, class_idx, x, y], 1.0)
        patch = gausses[batch_idx, class_idx, x - 1 : x + 2, y - 1 : y + 2]
        assert torch.all(patch <= 1.0) and torch.all(patch > 0.0)
        print(patch)


def test_embedding_placement_multi_channel():
    BATCH_SIZE = 2
    CLASSES = 5
    SPATIAL = 13, 17
    NUM_OBJECTS = 4
    EMBEDDING_DIM = 3

    locations = torch.from_numpy(
        np.random.randint(
            low=0,
            high=[BATCH_SIZE, CLASSES, SPATIAL[0], SPATIAL[1]],
            size=[NUM_OBJECTS, 4],
        )
    )
    embeddings = torch.from_numpy(
        np.random.randn(
            NUM_OBJECTS,
            EMBEDDING_DIM,
        ).astype("float32")
    )

    result = place_embeddings(
        center_locations=locations,
        embeddings=embeddings,
        shape=TensorShape(BATCH_SIZE, CLASSES, SPATIAL[0], SPATIAL[1]),
    )

    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == EMBEDDING_DIM
    assert result.size(2) == SPATIAL[0]
    assert result.size(3) == SPATIAL[1]

    for (batch_idx, _, x, y), embedding in zip(locations, embeddings):
        assert torch.allclose(result[batch_idx, :, x, y], embedding)


def test_corner_conversion_from_relative_to_absolute_and_back():
    N = 30
    WH = np.clip(
        np.random.normal(loc=0.5, scale=0.3, size=(N, 2)), a_min=0.0, a_max=0.95
    )
    CORNER_X0Y0 = np.random.uniform(0.0, WH, size=(N, 2))
    CORNER_X1Y1 = CORNER_X0Y0 + WH
    CENTERS = (CORNER_X0Y0 + CORNER_X1Y1) / 2.0

    CORNERS_ABSOLUTE = np.concatenate([CORNER_X0Y0, CORNER_X1Y1], axis=1)

    as_relative = generate_relative_corner_position(CORNERS_ABSOLUTE, CENTERS)
    np.testing.assert_allclose(as_relative[:, :2], -as_relative[:, 2:4])
    np.testing.assert_allclose(as_relative[:, 2:4] - as_relative[:, :2], WH)

    as_absolute = restore_corner_absolute_position(as_relative, CENTERS)
    np.testing.assert_allclose(CORNERS_ABSOLUTE, as_absolute)


if __name__ == "__main__":
    pytest.main()
