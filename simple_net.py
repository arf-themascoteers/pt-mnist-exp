import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleNet(nn.Module):
    def __init__(self, filters):
        super(SimpleNet, self).__init__()
        self.filters = torch.nn.Parameter(filters)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.filters.requires_grad = True

    def forward(self, x):
        similarity = torch.zeros((x.shape[0],self.filters.shape[0]))
        diff = torch.zeros_like(similarity)
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                similarity[i][j] = torch.sum(x[i] * self.filters[j])
                diff[i][j] = torch.sum(torch.abs(x[i] - self.filters[j])) / 5
        score = similarity - diff
        return F.log_softmax(score)