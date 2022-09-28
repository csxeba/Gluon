import dataclasses
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclasses.dataclass
class MetricIO:
    predictions: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    ground_truths: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    metrics: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    aggregated_metric: Optional[torch.Tensor] = None


class Metric(nn.Module):

    name: str = ""

    def __init__(self, display_name: str = ""):
        super().__init__()
        if not display_name:
            display_name = self.__class__.name
        if not display_name:
            raise ValueError(
                "Either set cls.name or pass display_name as a constructor parameter."
            )
        self.name = display_name

    def forward(self, metric_io: MetricIO) -> torch.Tensor:
        raise NotImplementedError


class Criteria(nn.Module):
    def __init__(
        self,
        metric_functions: List[Metric],
        aggregation_function: Optional[Metric] = None,
    ):
        """
        :param metric_functions: List[Metric]
            List of loss or other metric functions.
            Will be called in order. Later functions can access previous results.
        :param aggregation_function: Optional[Metric]
            Optionally produces an aggregated output from previously calculated losses/metrics
        """
        super().__init__()
        self.metric_functions = metric_functions
        self.aggregation_function = aggregation_function

    def forward(self, metric_io: MetricIO) -> MetricIO:
        for metric_function in self.metric_functions:
            metric_io.metrics[metric_function.name] = metric_function(metric_io)
        if self.aggregation_function is not None:
            metric_io.aggregated_metric = self.aggregation_function(metric_io)
        return metric_io
