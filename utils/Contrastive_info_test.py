import torch
import torch.nn.functional as F
from torch import nn


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, mask=None):

        # Normalize to unit vectors
        # 归一化
        query = F.normalize(query)
        positive_key = F.normalize(positive_key)
        # 归一化
        negative_keys = F.normalize(negative_keys)


        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        negative_logits =  torch.matmul(query, negative_keys.t())
        negative_logits = torch.gather(negative_logits, dim=1, index=mask)


          # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)

        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        loss= F.cross_entropy(logits / self.temperature, labels, reduction=self.reduction)
        return loss



if __name__ == '__main__':



    loss =InfoNCE()
    batch_size, num_negative, embedding_size = 32, 48, 128
    query = torch.randn(batch_size, embedding_size)
    positive_key = torch.randn(batch_size, embedding_size)
    negative_keys = torch.randn(num_negative, embedding_size)
    mask = torch.zeros(32,128)
    mask[:, :50] = 1.0

    output = loss(query, positive_key, negative_keys, mask )

    print(output)
