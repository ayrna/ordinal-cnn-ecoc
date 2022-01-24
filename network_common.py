from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from spacecutter.losses import CumulativeLinkLoss as OriginalCumulativeLinkLoss
from torch import Tensor, nn


__all__ = ['ExperimentModel', 'ordinal_ecoc_distance_loss', 'CumulativeLinkLoss']

class ExperimentModel(nn.Module, metaclass=ABCMeta):
    features: nn.Module
    avgpool: nn.Module
    classifier: nn.Module

    @abstractmethod
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        x = self.scores(x)
        probas = F.softmax(x, dim=1)
        labels = probas.argmax(dim=1)
        labels, probas = map(lambda t: t.detach().cpu().numpy(), (labels, probas))
        return labels, probas

    def non_regularized_parameters(self) -> List[nn.parameter.Parameter]:
        return list(set(self.parameters()) - set(self.regularized_parameters()))

    @abstractmethod
    def regularized_parameters(self) -> List[nn.parameter.Parameter]:
        pass

    @abstractmethod
    def on_batch_end(self):
        pass


def ordinal_ecoc_distance_loss(n_classes: int, device, class_weights: Optional[torch.Tensor] = None):
    target_class = np.ones((n_classes, n_classes-1), dtype=np.float32)
    target_class[np.triu_indices(n_classes, 0, n_classes-1)] = 0.0
    target_class = torch.tensor(target_class, dtype=torch.float32, device=device, requires_grad=False)

    if class_weights is not None:
        assert class_weights.shape == (n_classes,)
        class_weights = class_weights.float().to(device)
        mse = nn.MSELoss(reduction='none')

        def _weighted_ordinal_distance_loss(net_output, target):
            target_vector = target_class[target]
            weights = class_weights[target]  # type: ignore
            return (mse(net_output, target_vector).sum(dim=1) * weights).sum()

        return _weighted_ordinal_distance_loss
    else:
        mse = nn.MSELoss(reduction='sum')

        def _ordinal_distance_loss(net_output, target):
            target_vector = target_class[target]
            return mse(net_output, target_vector)

        return _ordinal_distance_loss


class CumulativeLinkLoss(OriginalCumulativeLinkLoss):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true = torch.unsqueeze(y_true, 1)
        return super(CumulativeLinkLoss, self).forward(y_pred, y_true)


_QWK_LOSS_EPSILON = 1e-9

class QWKLoss(nn.Module):
    def __init__(self, num_classes: int, weight: Optional[Tensor] =None, **kwargs) -> None:
        super().__init__(**kwargs)

        # Create cost matrix and register as buffer
        cost_matrix = torch.tensor(np.reshape(np.tile(range(num_classes), num_classes), (num_classes, num_classes))).float()
        cost_matrix = (cost_matrix - torch.transpose(cost_matrix, 0, 1)) ** 2
        
        if weight is not None:
            cost_matrix = cost_matrix.to(weight.device) * (1. / weight)
        
        self.register_buffer("cost_matrix", cost_matrix)

        self.num_classes = num_classes

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        output = F.softmax(output, dim=1)
        
        costs = self.cost_matrix[target]  # type: ignore

        numerator = costs * output
        numerator = torch.sum(numerator)

        sum_prob = torch.sum(output, dim=0)
        target_prob = F.one_hot(target, self.num_classes)
        n = torch.sum(target_prob, dim=0)

        denominator = ((self.cost_matrix * sum_prob[None, :]).sum(dim=1) * (n/n.sum())).sum()
        denominator = denominator + _QWK_LOSS_EPSILON

        return torch.log(numerator / denominator)


class OrdinalFullyConnected(nn.Module):
    classifiers: nn.ModuleList
    
    def __init__(self, input_size: int, num_classes: int):
        super(OrdinalFullyConnected, self).__init__()
        self.classifiers = nn.ModuleList(
            [nn.Linear(input_size, 1) for _ in range(num_classes-1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = [classifier(x) for classifier in self.classifiers]
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x
