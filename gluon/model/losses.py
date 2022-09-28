import torch

from gluon.interfaces.criteria import Metric, MetricIO


class FocalLoss(Metric):

    name = "focal_loss"

    def forward(self, metric_io: MetricIO) -> torch.Tensor:
        gt = metric_io.ground_truths["heatmap"]
        pred = torch.sigmoid(metric_io.predictions["heatmap"])

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class L1Loss(Metric):

    def __init__(self, tensor_key: str):
        super().__init__(display_name="l1_loss")
        self.tensor_key = tensor_key
        self.operation = torch.nn.L1Loss()

    def forward(self, metric_io: MetricIO) -> torch.Tensor:
        return self.operation.forward(
            metric_io.predictions[self.tensor_key],
            metric_io.ground_truths[self.tensor_key],
        )


class SumAggregator(Metric):

    def forward(self, metric_io: MetricIO) -> torch.Tensor:
        return sum(metric_io.metrics.values())
