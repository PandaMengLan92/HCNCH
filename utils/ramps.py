import numpy as np
import math
from torch.optim.lr_scheduler import _LRScheduler

def pseudo_rampup(T1, T2):
    def warpper(epoch):
        if epoch > T1:
            alpha = (epoch-T1) / (T2-T1)
            if epoch > T2:
                alpha = 1.0
        else:
            alpha = 0.0
        return alpha
    return warpper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""
    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0
    return warpper


def exp_rampdown(rampdown_length, num_epochs):
    """Exponential rampdown from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5* (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0
    return warpper


def cosine_rampdown(rampdown_length, num_epochs):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5* (epoch - (num_epochs - rampdown_length))
            return float(.5 * (np.cos(np.pi * ep / rampdown_length) + 1))
        else:
            return 1.0
    return warpper


def cosine_rampup(rampup_length, num_epochs):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch <= rampup_length:
            return float((epoch + 0.01) / rampup_length)
        elif epoch < 275:
            return float(0.5 * (math.cos((epoch - rampup_length) / (num_epochs- rampup_length) * math.pi) + 1))
        else:
            return 0.02
    return warpper

def cosine_warmup(rampup_length, num_epochs, ):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""


    def warpper(epoch):
        if epoch <= rampup_length:
            return float((epoch + 0.01) / rampup_length)
        elif epoch < 275:
            return float(0.5 * (math.cos((epoch - rampup_length) / (num_epochs- rampup_length) * math.pi) + 1))
        else:
            return 0.02
    return warpper



def exp_warmup(rampup_length, rampdown_length, num_epochs):
    rampup = exp_rampup(rampup_length)
    rampdown = exp_rampdown(rampdown_length, num_epochs)
    def warpper(epoch):
        return rampup(epoch)*rampdown(epoch)
    return warpper


def test_warmup():
    warmup = exp_warmup(80, 50, 500)
    for ep in range(500):
        print(warmup(ep))


class WarmupCosineLrScheduler(_LRScheduler):


    def __init__(self, optimizer, T_warmup, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_warmup =T_warmup
        self.last_epoch =last_epoch
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.base_lrs[0]
        if self.last_epoch < self.T_warmup:
            return [(self.last_epoch+0.0001)*lr / self.T_warmup]
        else:
            return [self.eta_min + (lr - self.eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2]

