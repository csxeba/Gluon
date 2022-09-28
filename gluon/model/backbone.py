from typing import Optional, Protocol

from torch import nn


class TorchVisionModelFactory(Protocol):
    def __call__(
        self,
        *,
        weights: Optional[str] = None,
        progress: bool = True,
        **kwargs,
    ) -> nn.Module:
        pass
