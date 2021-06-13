import torch
import torch.nn as nn


class SVD(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, rate_min=0, rate_max=5, rate_mean=0, dropout=0.0):
        super(SVD, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.rate_mean = rate_mean
        self.embed_dim = embed_dim

        self.rate_max = rate_max
        self.rate_min = rate_min

        self.embed_user = nn.Embedding(self.num_users, self.embed_dim)
        self.embed_item = nn.Embedding(self.num_items, self.embed_dim)

        self.bias_user = nn.Embedding(self.num_users, 1)
        self.bias_item = nn.Embedding(self.num_items, 1)

        nn.init.xavier_uniform_(self.embed_user.weight)
        nn.init.xavier_uniform_(self.embed_item.weight)

        nn.init.zeros_(self.bias_user.weight)
        nn.init.zeros_(self.bias_item.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, user, item):
        embed_u = self.embed_user(user)
        embed_v = self.embed_item(item)
        bias_u = self.bias_user(user)
        bias_v = self.bias_item(item)

        # u=(B, dim) v=(B, dim)  torch.dot: u*v 对于点位相乘，不求和， matmul：才是矩阵相乘
        predicted = (embed_u * embed_v).sum(1, keepdim=True) + bias_u + bias_v + self.rate_mean
        predicted = torch.clamp(predicted, self.rate_min, self.rate_max)

        return predicted.squeeze()


if __name__ == "__main__":
    model = SVD(num_users=100000, num_items=10000, embed_dim=64)
    a = torch.ones(4, 10, 2)
    b = torch.ones(4, 4, 2)
    c = torch.ones(4, 10, 1)
    # print(a)
    # s = torch.matmul(a, b.T).sum(1, keepdim=True) + c + 1

    import numpy as np

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([10, 20, 30])
    score = model.forward(x, y)
    print(score.shape)
