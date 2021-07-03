from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def plot(img):
    plt.imshow(img)
    plt.show()


def diff(y, f):
    return torch.sum(torch.abs(y-f))


def similarity(y,f):
    return torch.sum(y * f)


def calculate_score(y, f):
    return torch.sum(y * f)*2 - diff(y,f)


def detect(data, filters):
    score = torch.zeros(filters.shape[0])
    for i in range(filters.shape[0]):
        score[i] = calculate_score(data, filters[i])

    return torch.argmax(score)


def test():
    working_set = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    filters = torch.load("filters.pt")
    for data, y_true in dataloader:
        data = data.reshape(28, 28)
        pred = detect(data, filters)
        if pred == y_true.item():
            correct += 1
        # else:
        #     print(y_true.item())
        total += 1
        if total%1000 == 0:
            print(f"{total} tested. {correct} correct")
    print(f"{correct} correct among {total}")


def print_filters():
    filters = torch.load("filters.pt")
    for i in filters:
        for j in i:
            plt.imshow(j)
            plt.show()

if __name__ == "__main__":
    #print_filters()
    test()