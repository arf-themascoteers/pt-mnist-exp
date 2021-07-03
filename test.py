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

    new_score = torch.zeros(filters.shape[0])
    all_score = torch.sum(score)
    for i in range(filters.shape[0]):
        others_score = all_score - score
        new_score = score - (others_score/10)

    return torch.argmax(new_score)


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
        else:
            print(y_true.item())
        total += 1

    print(f"{correct} correct among {total}")


def print_filters():
    filters = torch.load("filters.pt")
    for i in filters:
        plt.imshow(i)
        plt.show()

if __name__ == "__main__":
    test()