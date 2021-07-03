from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def train():
    NUM_EPOCHS = 3
    BATCH_SIZE = 1000

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=1, shuffle=False)
    filters = torch.zeros([10,28,28])
    n = torch.zeros(10)
    i = 0
    for data,y_true in dataloader:
        data = data.reshape(28,28) * 255
        filters[y_true] = ((n[y_true] / (n[y_true] + 1)) * filters[y_true]) + (data/(n[y_true] + 1))
        n[y_true] += 1
        # if i%100 == 0:
        #     print(y_true)
            #plt.imshow(filters[y_true].numpy().reshape(28,28))
            #plt.show()
    filters = filters/255
    torch.save(filters,"filters.pt")



