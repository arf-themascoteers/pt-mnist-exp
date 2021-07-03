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

if __name__ == "__main__":
    train()