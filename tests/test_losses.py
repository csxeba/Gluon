import pytest
import torch

from gluon.interfaces import criteria as gloss


class _TestLoss1(gloss.Metric):
    name = "first_loss"

    def forward(self, metric_io: gloss.MetricIO) -> torch.Tensor:
        return metric_io.predictions["first"] + metric_io.ground_truths["first"]


class _TestLoss2(gloss.Metric):
    name = "second_loss"

    def forward(self, metric_io: gloss.MetricIO) -> torch.Tensor:
        return torch.tensor(10)


class _TestAggregatorSelectFirst(gloss.Metric):
    name = "select_first"

    def forward(self, metric_io: gloss.MetricIO) -> torch.Tensor:
        return metric_io.metrics["first_loss"]


class _TestAggregatorSum(gloss.Metric):
    name = "aggregate_sum"

    def forward(self, metric_io: gloss.MetricIO) -> torch.Tensor:
        return torch.sum(torch.tensor(list(metric_io.metrics.values())))


def test_criteria_with_select_first():

    pred = {"first": torch.tensor(1), "second": torch.tensor(1000)}
    gt = {"first": torch.tensor(10), "second": torch.tensor(314)}

    criteria = gloss.Criteria(
        metric_functions=[_TestLoss1(), _TestLoss2()],
        aggregation_function=_TestAggregatorSelectFirst(),
    )

    metric_io = gloss.MetricIO(
        predictions=pred,
        ground_truths=gt,
    )
    criteria(metric_io)
    assert metric_io.metrics["first_loss"] == 11
    assert metric_io.metrics["second_loss"] == 10
    assert metric_io.aggregated_metric is not None
    assert metric_io.aggregated_metric == 11


def test_criteria_with_sum():
    metric_io = gloss.MetricIO(
        predictions={"first": torch.tensor(1), "second": torch.tensor(1000)},
        ground_truths={"first": torch.tensor(10), "second": torch.tensor(314)},
    )
    criteria_sum = gloss.Criteria(
        metric_functions=[_TestLoss1(), _TestLoss2()],
        aggregation_function=_TestAggregatorSum(),
    )
    criteria_sum(metric_io)
    assert metric_io.metrics["first_loss"] == 11
    assert metric_io.metrics["second_loss"] == 10
    assert metric_io.aggregated_metric is not None
    assert metric_io.aggregated_metric == 21


if __name__ == "__main__":
    pytest.main()
