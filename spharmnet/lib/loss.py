"""
August 2022

Ilwoo Lyu, ilwoolyu@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    def __init__(self, eps=0, weights=None, ignore_index=None):

        """
        Dice loss.

        Notes
        _____
        https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
        """

        super().__init__()

        self.eps = eps
        self.weights = weights if weights is not None else 1
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = F.softmax(input, 1)
        encoded_target = input.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = input * encoded_target
        numerator = 2 * intersection.sum(0).sum(1)
        denominator = input + encoded_target

        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1) + self.eps
        loss_per_channel = self.weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / input.size(1)
