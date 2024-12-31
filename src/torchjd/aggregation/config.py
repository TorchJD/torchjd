from .bases import Aggregator
from conflictfree.grad_operator import ConFIGOperator
from conflictfree.momentum_operator import PseudoMomentumOperator
from conflictfree.weight_model import EqualWeight, WeightModel
from conflictfree.length_model import ProjectionLength, LengthModel
from conflictfree.loss_recorder import LossRecorder
from typing import Optional

class ConFIGAggregator(Aggregator):

    def __init__(self,
                 weight_model: WeightModel = EqualWeight(),
                 length_model: LengthModel = ProjectionLength(),
                 allow_simplified_model: bool = True,
                 use_latest_square: bool = True):
        super().__init__()
        self._config_operator = ConFIGOperator(
            weight_model=weight_model,
            length_model=length_model,
            allow_simplified_model=allow_simplified_model,
            use_latest_square=use_latest_square)    

    def forward(self, matrix):
        return self._config_operator.calculate_gradient(matrix)
    