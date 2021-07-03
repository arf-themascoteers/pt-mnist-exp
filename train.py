from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def train():
    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=1, shuffle=False)
    filters = torch.zeros([10,28,28])
    count = torch.zeros(10)
    for data,y in dataloader:
        data = data.reshape(28,28) * 255
        filters[y] = ((count[y] / (count[y] + 1)) * filters[y]) + (data/(count[y] + 1))
        count[y] += 1
    filters = filters/255
    torch.save(filters,"filters.pt")



