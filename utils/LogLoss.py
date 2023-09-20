import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# 两两样本计算 https://blog.csdn.net/weixin_42764932/article/details/112998284

class PairLoss(nn.Module):
    def __init__(self):
        super(PairLoss, self).__init__()

    def forward(self, S, U):
        theta = U @ U.t() / 2
        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)
        pair_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        regular_term = (U - U.sign()).pow(2).mean()
        
        return pair_loss + regular_term * 0.05
        #return pair_loss

class ConsistenceLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super(ConsistenceLoss, self).__init__()

        self.ConLoss = torch.nn.BCEWithLogitsLoss()
        self.temperature = temperature

    def forward(self, U, Un, M=None):
        # 有标签和锚点内积， 锚点相似性矩阵[0,1] 锚点的权重[-1,1] 有标签的掩码
        theta = torch.mul(U, Un).sum(1) / self.temperature
        # Prevent exp overflow
        theta = torch.clamp(theta, min=-100, max=50)
        label = torch.ones((theta.size())).cuda()
        loss = self.ConLoss(theta, label)

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=2.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature1, feature2, labels=None, positive_mask=None, weight_mask=None):
        """label"""
        """feature"""
        # 归一化
        #feature1 = F.normalize(feature1)
        #feature2 = F.normalize(feature2)
        # compute logits#余弦 但是加入一个超参，不保持余弦值为1 这就成了有标记和无标记的内积
        #sim = torch.div(torch.matmul(feature1, feature2.t()), self.temperature)i
        #2.5
        sim = torch.matmul(feature1, feature2.t())/self.temperature

        # for numerical stability
        ##对角线相似性为0 因为 对角线元素不需要计算
        """
        #这个最重要的是正例mask 和 负例mask
        mask
        文中的正例和负例
        对于每一个无标签数据来说和多个有标记数据组合， 然后有正例和负例
        如果近邻样本来自同一类别，可靠
        近邻样本来自不同类别，不可靠 打散（对比学习）

        """
        # tile mask #这个mask 要根据有标记和无标记样本的关系进行计算一个相似性mask
        # 一方面使用无标记数据和有标记数据的预测的相似性，一方面使用最像

        # 全1矩阵，但是对角线为0 排除自身相似性 这个是分母的部分。
        # compute log_prob

        exp_logits = torch.exp(sim)
        #log_prob = (sim + 0.9*torch.log(positive_mask.sum(1))) - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = -(weight_mask * log_prob).sum(1) / positive_mask.sum(1)
        loss = mean_log_prob_pos.mean()

        return loss


class image_wise_Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=2.5, device=None):
        super(image_wise_Loss, self).__init__()
        self.temperature = temperature
        self.device= device

    def forward(self, feature1, feature2):

        batch_size = feature1.size(0)
        features = torch.cat([feature1, feature2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        anchor_dot_contrast = torch.matmul(features, features.t())/self.temperature
        mask = mask.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(self.device),
            0
        )

        mask = mask * logits_mask
        logits = torch.exp(anchor_dot_contrast)
        pos= (logits*mask).sum(1, keepdim=True)

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        neg= exp_logits.sum(1, keepdim=True)
        log_prob =  - torch.log(pos/neg)  # 出
        loss = log_prob.mean()
        return loss

class SoftSimilarity(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, code_length):
        super(SoftSimilarity, self).__init__()
        self.code_length =code_length


    def forward(self, soft_similarity, U):
        theta = U @ U.t()

        reg = (theta - self.code_length * soft_similarity) ** 2
        loss = reg.mean()

        return loss


class SoftSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=2.5, device=None ):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device


    def forward(self, f1, f2, max_probs, labels=None, mask=None, select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        这是把自身的与同类的分开考虑的，应该在加入一个一起考虑的
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

        batch_size = f1.size(0)

        features = torch.cat([f1, f2], dim=0)
        labels = labels.contiguous().view(-1, 1)


        mask = torch.eq(labels, labels.t()).float()
            #max_probs = max_probs.reshape((batch_size,1))
        max_probs = max_probs.contiguous().view(-1, 1)
        score_mask = torch.matmul(max_probs, max_probs.t())
        #thresh_mask = (score_mask > 0.8)
        #mask_score = mask * score_mask * thresh_mask
        mask_score = mask*score_mask

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        logits_mask= logits_mask.repeat(2, 2)


        anchor_dot_contrast = torch.matmul(features, features.t())/2.5
        # for numerical stability
        #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask_score = mask_score.repeat(2, 2)
        mask_score = mask_score * logits_mask



        # compute log_prob
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = -(mask_score * log_prob).sum(1) / mask.repeat(2, 2).sum(1)
        #mean_log_prob_pos = -(mask_score * log_prob).mean(dim=1)

        # loss
        #loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        #loss = loss.view(anchor_count, batch_size)

        #if reduction == "mean":
        loss = mean_log_prob_pos.mean()

        return loss

class SoftSupConLossV(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=2.5, device=None ):
        super(SoftSupConLossV, self).__init__()
        self.temperature = temperature
        self.device = device


    def forward(self, f1, f2, max_probs, labels=None, mask=None, select_matrix=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        这是把自身的与同类的分开考虑的，应该在加入一个一起考虑的
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

        batch_size = f1.size(0)

        features = torch.cat([f1, f2], dim=0)
        labels = labels.contiguous().view(-1, 1)


        mask = torch.eq(labels, labels.t()).float()
            #max_probs = max_probs.reshape((batch_size,1))
        max_probs = max_probs.contiguous().view(-1, 1)
        score_mask = torch.matmul(max_probs, max_probs.t())

        if select_matrix is not None:
            mask = mask * select_matrix

        mask_score = mask * score_mask

        mask_score += torch.eye(mask_score.shape[0]).to(self.device)
        mask_score[mask_score > 1] = 1

        mask_score = mask_score.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(mask_score),
            1,
            torch.arange(batch_size*2).view(-1, 1).to(self.device),
            0
        )

        anchor_dot_contrast = torch.matmul(features, features.t())/2.5
        # for numerical stability
        #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #logits = anchor_dot_contrast - logits_max.detach()

        # tile mask

        mask_score = mask_score * logits_mask
        mask_num = mask.repeat(2, 2) * logits_mask

        # compute log_prob
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = -(mask_score * log_prob).sum(1) / mask_num.sum(1)

        # loss
        #loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        #loss = loss.view(anchor_count, batch_size)

        #if reduction == "mean":
        loss = mean_log_prob_pos.mean()

        return loss

if __name__ == '__main__':
    batch_size = 4
    feature = torch.randn(batch_size * 2, 3)

    max_pro= torch.rand([4, 5])

    woutputs = F.softmax(max_pro, 1)
    wprobs, wpslab = woutputs.max(1)
    f1, f2 = torch.split(feature, [batch_size, batch_size], dim=0)
    conloss = SoftSupConLossV()
    loss=conloss(f1,f2,  )


