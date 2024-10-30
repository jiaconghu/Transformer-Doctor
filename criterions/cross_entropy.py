# Copyright (c) QIU Tian. All rights reserved.

from typing import List, Dict

import torch.nn.functional as F

from utils.misc import accuracy

from torch import nn


class BaseCriterion(nn.Module):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict

    def forward(self, outputs, targets, **kwargs):
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, **kwargs))


class CrossEntropy(BaseCriterion):
    def __init__(self, losses: List[str], weight_dict: Dict[str, float]):
        super().__init__(losses, weight_dict)

    def loss_labels(self, outputs, targets, **kwargs):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'loss_labels(self, outputs, targets, **kwargs)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs["logits"]

        loss_ce = F.cross_entropy(outputs, targets, reduction='mean')
        losses = {'loss_ce': loss_ce, 'class_error': 100 - accuracy(outputs, targets)[0]}

        return losses
