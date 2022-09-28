from typing import Dict, List, Optional

import torch
import tqdm  # type: ignore[import]
from torch.utils.data import DataLoader

from gluon.codec.method import ModelCodec
from gluon.interfaces.criteria import Criteria, MetricIO
from gluon.interfaces.data import GluonDataPoint, GluonLabels
from gluon.interfaces.network import ModelInterface


class ExecutionBase:
    def __init__(self, loader: DataLoader, model: ModelInterface, codec: ModelCodec):
        self.loader = loader
        self.model = model
        self.codec = codec

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Inference(ExecutionBase):
    def run(self):
        i: int
        sample_batch: List[GluonDataPoint]
        detections: List[List[GluonLabels]] = []
        for i, sample_batch in enumerate(self.loader):
            encoded_image = self.codec.encode_inputs(
                [sample.inputs for sample in sample_batch]
            )
            prediction = self.model(encoded_image)
            decoded_prediction = self.codec.decode_labels(prediction)
            detections.append(decoded_prediction)


class Training(ExecutionBase):
    def __init__(
        self,
        loader: DataLoader,
        model: ModelInterface,
        codec: ModelCodec,
        training_criteria: Criteria,
        optimizer: torch.optim.Optimizer,
        more_metrics: Optional[Dict[str, Criteria]] = None,
    ):
        super().__init__(loader, model, codec)
        self.training_criteria = training_criteria
        self.more_metrics = more_metrics
        self.optimizer = optimizer

    def run(self, epochs: int):
        sample_batch: List[GluonDataPoint]
        aggregated_metrics = []
        for epoch in tqdm.trange(1, epochs + 1):
            progress_bar = tqdm.tqdm(
                self.loader, total=len(self.loader), unit="batch", postfix=" "
            )
            for sample_batch in progress_bar:
                enc_labels = self.codec.encode_labels(
                    [sample.labels for sample in sample_batch]
                )
                enc_inputs = self.codec.encode_inputs(
                    [sample.inputs for sample in sample_batch]
                )
                prediction = self.model(enc_inputs)
                metric_io = MetricIO(prediction, enc_labels)
                self.training_criteria.forward(metric_io)
                if metric_io.aggregated_metric is None and len(metric_io.metrics) > 1:
                    raise RuntimeError(
                        "Metrics should be aggregated in the training_criteria."
                    )
                elif (
                    metric_io.aggregated_metric is None and len(metric_io.metrics) == 1
                ):
                    metric_name, aggregated_metric = metric_io.metrics.popitem()
                    metric_io.metrics[metric_name] = aggregated_metric
                elif metric_io.aggregated_metric is not None:
                    aggregated_metric = metric_io.aggregated_metric
                else:
                    assert False

                self.optimizer.zero_grad()
                aggregated_metric.backward()
                self.optimizer.step()

                aggregated_metrics.append(aggregated_metric.detach().item())

                progress_bar.set_postfix_str(f" Loss: {aggregated_metric:.4f}")
