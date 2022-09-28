import numpy as np
import pytest
import torch

from gluon.interfaces.data import Location, TensorShape


def test_tensorshape_instantiation():
    BATCH_SIZE = 3
    NUM_CLASSES = 10
    SPATIAL_X = 64
    SPATIAL_Y = 32

    SHAPE = BATCH_SIZE, NUM_CLASSES, SPATIAL_X, SPATIAL_Y

    tensor_shape = TensorShape(
        batch_size=BATCH_SIZE,
        depth=NUM_CLASSES,
        width=SPATIAL_X,
        height=SPATIAL_Y,
    )

    tensor_shape_tup = tensor_shape.as_tuple()
    assert tensor_shape_tup == SHAPE


def test_tensorshape_from_tensor_factory():
    BATCH_SIZE = 3
    NUM_CLASSES = 10
    SPATIAL_X = 64
    SPATIAL_Y = 32

    SHAPE = BATCH_SIZE, NUM_CLASSES, SPATIAL_X, SPATIAL_Y

    tensor = torch.empty(SHAPE, dtype=torch.uint8)

    tensor_shape_from_tensor = TensorShape.from_tensor(tensor)
    tensor_shape_from_tensor_tup = tensor_shape_from_tensor.as_tuple()
    assert tensor_shape_from_tensor_tup == SHAPE


def test_location_instantiation():
    NUM_CLASSES = 10
    SPATIAL_X = 64
    SPATIAL_Y = 32
    NUM_OBJECTS_PER_BATCH = [5, 2, 9]

    object_coords_per_batch = []
    for batch_idx, num_objects in enumerate(NUM_OBJECTS_PER_BATCH):
        coords = np.random.randint(
            low=0,
            high=(NUM_CLASSES, SPATIAL_X, SPATIAL_Y),
            size=(num_objects, 3),
            dtype=int,
        )
        batch_idx = np.full(num_objects, fill_value=batch_idx, dtype=int)
        object_coords_per_batch.append(
            np.concatenate([batch_idx[:, None], coords], axis=1)
        )

    object_coords_flat = np.concatenate(object_coords_per_batch, axis=0)
    location = Location(
        center_xy=torch.tensor(
            object_coords_flat[:, 2:4] / np.array([SPATIAL_X, SPATIAL_Y]),
            dtype=torch.float32,
        ),
        class_ids=torch.tensor(object_coords_flat[:, 1]),
        batch_idx=torch.tensor(object_coords_flat[:, 0]),
    )

    as_tensor_indices = location.as_tensor_indices(
        TensorShape(
            batch_size=len(NUM_OBJECTS_PER_BATCH),
            depth=NUM_CLASSES,
            width=SPATIAL_X,
            height=SPATIAL_Y,
        )
    )

    np.testing.assert_equal(object_coords_flat, as_tensor_indices)


if __name__ == "__main__":
    pytest.main()
