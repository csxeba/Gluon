import torch


def assert_vector_of_embeddings_equality_unordered(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
):
    assert vec1.size() == vec2.size()
    D = torch.abs(vec1[None, :, :] - vec2[:, None, :]).sum(dim=-1)
    has_a_near_zero_pair = torch.any(D < 1e-3, dim=0)
    assert torch.all(has_a_near_zero_pair)

