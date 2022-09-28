from typing import Tuple

import pytest
import torch

from gluon.transforms import ResizeForTraining


@pytest.mark.parametrize(
    ["input_shape", "target_shape"],
    [[(64, 64), (16, 16)],
     [(368, 656), (480, 480)],
     [(640, 320), (400, 400)]]
)
def test_resize_transform(
    input_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
):
    pytest.skip("Unfinished")
    input_tensor = torch.ones((1, 3) + input_shape)
    op = ResizeForTraining(input_element_key="x", target_size_wh=list(target_shape))

    gdp = GluonDataPoint()

    output_tensor = op.forward(input_tensor)

if __name__ == "__main__":
    pytest.main()
