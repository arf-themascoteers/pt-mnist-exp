from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from simple_net import SimpleNet

def plot(img):
    plt.imshow(img)
    plt.show()


def diff(y, f):
    return torch.sum(torch.abs(y-f))/5


def similarity(y,f):
    return torch.sum(y * f)


def calculate_score(y, f):
    sim = similarity(y,f)
    dif = diff(y,f)
    return sim - dif


def detect(data, filters):
    score = torch.zeros(filters.shape[0])
    for i in range(filters.shape[0]):
        score[i] = calculate_score(data, filters[i])

    return torch.argmax(score), score


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
        pred, score = detect(data, filters)
        if pred == y_true.item():
            correct += 1
        else:
            y_true.item()
        total += 1
        if total%1000 == 0:
            print(f"{total} tested. {correct} correct")
    print(f"{correct} correct among {total}")


def test_machine():
    BATCH_SIZE = 100

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    filters = torch.load("filters.pt")
    model = SimpleNet(filters)
    model.load_state_dict(torch.load("models/machine.h5"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1

    print(f"{correct} correct among {len(working_set)}")


def print_filters():
    filters = torch.load("filters.pt")
    for i in filters:
        for j in i:
            plt.imshow(j)
            plt.show()

if __name__ == "__main__":
    #print_filters()
    #test()
    test_machine()