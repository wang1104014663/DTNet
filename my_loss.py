import warnings
import torch
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import LogSoftmax
from torch.nn import functional as F


class _Loss(Module):
    def __init__(self, size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce


class TripletMarginLoss(_Loss):
    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False, size_average=True, reduce=True):
        super(TripletMarginLoss, self).__init__(size_average, reduce)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin, self.p,
                                     self.eps, self.swap, self.size_average, self.reduce)