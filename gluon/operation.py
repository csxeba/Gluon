from typing import Generic, Tuple, TypeVar

import numpy as np
import torch

from gluon.interfaces import data

__all__ = [
    "gauss_1D",
    "gauss_2D",
    "place_embeddings",
    "generate_relative_corner_position",
    "restore_corner_absolute_position",
    "snap_coordinates_to_grid",
]


def gauss_1D(
    center_coords_1D: torch.Tensor,
    sigma: torch.Tensor,
    shape: int,
    min_threshold: float,
) -> torch.Tensor:
    """
    :param center_coords_1D: Tensor[n]
        dim0 corresponds to all the object instances in a batch.
    :param sigma: Tensor[n]
        Standard deviation to be applied to the gaussian blobs. For all object instances.
    :param shape: int
        Describes the shape of the output.
    :param min_threshold: float
        Gaussian values under min_threshold will be set equal to 0.
    :return: Tensor[num_classes, spatial]
    """
    rangex = torch.arange(shape).double()[None, :]  # shape: [1, spatial]
    center_coords = center_coords_1D[:, None]  # shape: [n, 1]

    diff = (rangex - center_coords) / sigma[:, None]  # shape: [n, spatial]
    exp = torch.exp(torch.pow(diff, 2.0) * -0.5)  # shape: [n, spatial]
    return exp * torch.gt(exp, min_threshold).double()


def gauss_2D(
    center_locations: torch.Tensor,
    sigma: torch.Tensor,
    tensor_shape: data.TensorShape,
    min_threshold: float,
) -> torch.Tensor:
    """
    :param center_locations: Tensor[n, 4]
        dim0 corresponds to all the object instances in a batch.
        dim1 corresponds to [batch_index, class_index, spatial_coord_x, spatial_coord_y].
    :param sigma: Tensor[n]
        Standard deviation to be applied to the gaussian blobs. For all object instances.
    :param tensor_shape: TensorShape[int, int, int, int]
        Describes the shape of the output. Dims correspond to [batch_size, num_classes, spatial_w, spatial_h]
    :param min_threshold: float
        Gaussian values under min_threshold will be set equal to 0.
    :return:
    """
    shape = tensor_shape.as_tuple()
    output_tensor = torch.zeros(shape)
    shape_x, shape_y = shape[2], shape[3]
    for batch_id in range(shape[0]):
        batch_mask = center_locations[:, 0] == batch_id
        for class_id in range(shape[1]):
            class_mask = center_locations[:, 1] == class_id
            batch_and_class_mask = torch.logical_and(batch_mask, class_mask)
            batch_and_class_coords = center_locations[batch_and_class_mask]
            batch_and_class_sigma = sigma[batch_and_class_mask]
            if batch_and_class_coords.size(0) == 0:
                continue
            gauss_x = gauss_1D(
                center_coords_1D=batch_and_class_coords[:, 2],
                sigma=batch_and_class_sigma,
                shape=shape_x,
                min_threshold=min_threshold,
            )  # [n, x]
            gauss_y = gauss_1D(
                center_coords_1D=batch_and_class_coords[:, 3],
                sigma=batch_and_class_sigma,
                shape=shape_y,
                min_threshold=min_threshold,
            )  # [n, y]
            outer_prod_gauss = torch.max(
                gauss_x[:, :, None] * gauss_y[:, None, :], dim=0
            ).values
            output_tensor[batch_id, class_id, :, :] = outer_prod_gauss
    return output_tensor


def place_embeddings(
    center_locations: torch.Tensor,
    embeddings: torch.Tensor,
    shape: data.TensorShape,
):
    """
    :param center_locations: torch.Tensor[N, 4]
        Dim0 corresponds to all objects of all classes in a batch
        Dim1 corresponds to [batch_index, class_index, coord_x, coord_y]
    :param embeddings: torch.Tensor[N, dim]
        Dim0 corresponds to all objects of all classes in a batch
        Dim1 holds the embedding value.
    :param shape: Tuple[int, int, int, int]
        Describes the output tensor shapes: [batch_size, num_classes, width, height]
    :return: torch.Tensor
        Embeddings placed on the appropriate tensor indices
    """

    batch_dim, _, width, heigth = shape.as_tuple()
    depth = embeddings.size(1)
    result = torch.zeros([batch_dim, depth, width, heigth], dtype=torch.float32)
    for (batch_idx, class_idx, coord_x, coord_y), embedding in zip(
        center_locations, embeddings
    ):
        result[batch_idx, :, coord_x, coord_y] = embedding

    return result


def generate_relative_corner_position(
    corners_absolute: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    centers_centers = np.concatenate([centers, centers], axis=1)
    corners_relative = corners_absolute - centers_centers
    return corners_relative


def restore_corner_absolute_position(
    corners_relative: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    centers_centers = np.concatenate([centers, centers], axis=1)
    corners_absolute = corners_relative + centers_centers
    return corners_absolute


def snap_coordinates_to_grid(
    coordinates: np.ndarray,
    grid_resolution: Tuple[int, int],
) -> np.ndarray:

    if len(coordinates) == 0:
        return coordinates

    assert 0.0 <= np.max(coordinates) < 1.0, f"Max: {np.max(coordinates)}"
    scaler = np.array(grid_resolution).astype(np.float32)
    coordinates_upscaled = coordinates * scaler[None, :]
    coordinates_upscaled_snapped = np.round(coordinates_upscaled)
    coordinates_low_scale_snapped = coordinates_upscaled_snapped / scaler[None, :]
    return coordinates_low_scale_snapped


def assert_doesnt_contain_center_collision(
    coordinates: np.ndarray,
    grid_resolution: Tuple[int, int],
):
    N = len(coordinates)
    if N == 0:
        return

    scaler = np.array(grid_resolution).astype(np.float32)
    coordinates_upscaled = coordinates * scaler[None, :]
    coordinates_upscaled_snapped = np.round(coordinates_upscaled).astype(int)
    N_unique = len(np.unique(coordinates_upscaled_snapped, axis=1))
    contains_collision = N_unique < N
    assert not contains_collision, (
        f"Coordinates contain {N - N_unique} collisions "
        f"@ resolution {grid_resolution[0]}x{grid_resolution[1]}"
    )
