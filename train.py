import time

from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from simple_net import SimpleNet
import torch.nn.functional as F

def pretrain():
    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    N_FILTER = 20
    dataloader = DataLoader(working_set, batch_size=1, shuffle=False)
    filters = torch.zeros([10,N_FILTER,28,28])
    count = torch.zeros(10,N_FILTER)
    next_index = torch.zeros(10, dtype=torch.int)
    for data,y in dataloader:
        y = y.item()
        current_index = next_index[y].item()
        data = data.reshape(28,28) * 255
        old_count = count[y][current_index]
        new_count = count[y][current_index] + 1
        current_filter = filters[y][current_index]
        previous_contribution = (old_count/new_count) * current_filter
        current_x_contribution = data/new_count
        filters[y][current_index] = previous_contribution + current_x_contribution
        count[y][current_index] += 1
        next_index[y] = (next_index[y] + 1) % N_FILTER
    filters = filters/255
    torch.save(filters,"filters.pt")


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 10
    BATCH_SIZE = 3000

    working_set = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    filters = torch.load("filters.pt").to(device)
    model = SimpleNet(filters).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    start = time.time()
    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            data = data.reshape(data.shape[0],28,28).to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            y_true = y_true.to(device)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        torch.save(model.state_dict(), 'models/machine.h5')
    end = time.time()
    print(f"Training time: {round(end-start)}")
    return model

if __name__ == "__main__":
    #pretrain()
    train()