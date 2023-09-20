import torch
import numpy as np
from scipy.spatial.distance import cdist,pdist,squareform

class cos_similrty():

    def consine_dis(self, a, b):
        dis = []
        for i in range(b.size(0)):
            # print(b[i].view(1,-1))
            attention_score = torch.cosine_similarity(a, b[i].view(1, -1))
            dis.append(np.array(attention_score))
        dis = torch.from_numpy(np.array(dis))

        return dis

    def consine_similarity(self, a, b):
        dis = []
        for i in range(a.size(0)):
            attention_score = torch.cosine_similarity(a[i].view(1, -1), b)
            dis.append(np.array(attention_score))
        dis = torch.from_numpy(np.array(dis))
        return dis

    def cosine_dist(self, x, y):

        xy = x.mm(y.t())
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        xy_norm = x_norm.mm(y_norm.t())
        return xy / xy_norm.add(1e-10)

    def consin(self, a):
        euc_ = pdist(a, 'cosine')

        euc_dis = squareform(euc_)
        orig_euc_dis = euc_dis
        S_real = 1 - orig_euc_dis
        print(S_real)

    def consin_pairwise(self, a, b):
        euc = cdist(a, b, 'cosine')
        S_real = 1 - euc
        return S_real


class Euclidean_similarity():

    def eucl_dist(self, x, y):
        """ Compute Pairwise (Squared Euclidean) Distance

        Input:
            x: embedding of size M x D
            y: embedding of size N x D

        Output:
            dist: pairwise distance of size M x N
        """
        x2 = torch.sum(x ** 2, dim=1, keepdim=True).expand(-1, y.size(0))
        y2 = torch.sum(y ** 2, dim=1, keepdim=True).t().expand(x.size(0), -1)
        xy = x.mm(y.t())
        s = x2 - 2 * xy + y2
        return s.sqrt_()

    def euclidean_dist(self,x,y):

        eu = cdist(x, y, metric='euclidean')


        return eu

if __name__ == '__main__':
    caculate_s = cos_similrty()
    eua = Euclidean_similarity()
    a = np.array([[1., 2., 3.], [4., 5., 6.],[6,-1,5],[3,9,-4]])
    b = np.array([[1., 2., -3.], [-1., -2., 3.], [4., 5., 6.],[4, 5, 6],[-1,-3,-6]])
    print(eua.euclidean_dist(a, b))

    a = torch.tensor(a)
    b = torch.tensor(b)
    sdf= eua.eucl_dist(a,b)
    # 欧式距离归一化
    sdf =torch.tensor(1.0) / (sdf+1e-7)
    s=torch.argsort(sdf, 1, descending=True)[:, 2:]
    print(s)
    sdf.scatter_(dim=1, index=s, src=torch.tensor(0.0))
    print(sdf)
    f= sdf / torch.sum(sdf,dim=1,keepdim=True)
    print(f)

    """
    print(caculate_s.consin_pairwise(a,b))
    a = torch.tensor(a)
    b = torch.tensor(b)
    print(caculate_s.cosine_dist(a, b))
    print(caculate_s.consine_similarity(a,b))
    print(caculate_s.consine_dis(a,b))
    """
