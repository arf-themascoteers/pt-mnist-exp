from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from simple_net import SimpleNet

def plot(img):
    plt.imshow(img)
    plt.show()


def explore():
    filters = torch.load("filters.pt")
    model = SimpleNet(filters)
    model.load_state_dict(torch.load("models/machine.h5"))
    model.eval()
    with torch.no_grad():
        for i in model.filters:
            plot(i[0].numpy())

if __name__ == "__main__":
    #print_filters()
    #test()
    explore()