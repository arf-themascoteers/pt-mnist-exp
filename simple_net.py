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
        x = x.reshape(x.shape[0],-1)
        filter_sum = self.filters.sum(dim=1)/self.filters.shape[1]
        filter_sum = filter_sum.reshape(self.filters.shape[0],-1)
        similarity = torch.mm(x,filter_sum.T)
        score = similarity
        return F.log_softmax(score)