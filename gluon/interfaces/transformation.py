from .data import GluonDataPoint


class Transformation:

    def __call__(self, data_point: GluonDataPoint) -> GluonDataPoint:
        return self.forward(data_point)

    def forward(self, data_point: GluonDataPoint) -> GluonDataPoint:
        raise NotImplementedError

    def reverse(self, data_point: GluonDataPoint) -> GluonDataPoint:
        raise NotImplementedError
