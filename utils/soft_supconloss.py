""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, base_temperature=0.07, device=None):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, max_probs, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]


        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(self.device)
            #max_probs = max_probs.reshape((batch_size,1))
        max_probs = max_probs.view(-1, 1)
        score_mask = torch.matmul(max_probs, max_probs.t())
        print(score_mask)

        mask = mask * score_mask

        # compute logits
        anchor_dot_contrast = torch.matmul(features, features.t()) / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(2, batch_size)


        loss = loss.mean()

        return loss


if __name__ == '__main__':
    batch_size = 4
    feature = torch.randn(batch_size * 2, 3)
    label = torch.tensor([[1, 0, 1, 2]])
    max_pro= torch.rand([5, 4])

    f1, f2 = torch.split(feature, [batch_size, batch_size], dim=0)
    f =torch.cat((f1,f2),dim=0)

    conloss = SoftSupConLoss()
    loss = conloss(f, max_pro, label)
