import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss as TorchMSE
from torch.nn import CrossEntropyLoss

__all__ = ["CELossOnList", "MSELossOnList", "MSELoss", "CrossEntropyLoss"]

class MSELoss(TorchMSE):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, pred, tgt):
        pred = pred.reshape(-1)
        return super().forward(pred, tgt)

class LossOnList(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, pred, tgt):
        loss = []

        if isinstance(tgt, torch.Tensor):
            for idx, p in enumerate(pred):
                loss.append(self.functional_loss(p, tgt[:, idx]))
        elif isinstance(tgt, list):
            for p, t in zip(pred, tgt):
                loss.append(self.functional_loss(p, t))

        return loss

class CELossOnList(LossOnList):
    def __init__(self, args):
        super().__init__(args)
        self.functional_loss = F.cross_entropy

class MSELossOnList(LossOnList):
    def __init__(self, args):
        super().__init__(args)
        self.functional_loss = F.mse_loss

    def forward(self, pred, tgt):
        pred = [p.reshape(-1) for p in pred]
        return super().forward(pred, tgt)
