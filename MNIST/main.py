import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Tuple


DATASET_DIR = './data' # path to download mnist dataset

TRAIN_DATASET = datasets.MNIST(DATASET_DIR,   # Dataset root path
                               train=True,    # Train data
                               download=True) # Download if not exist

TEST_DATASET = datasets.MNIST(DATASET_DIR,    # Dataset root path
                              train=False)    # Test data


class Network(nn.Module):
    """Simple Neural Network contains conv layer and fc layer
    """
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer) -> float:
    """Train function

        Arguments:
            model (nn.Module): some networks extends from ``nn.Module``.
            device (torch.device): device for use CUDA if available.
            train_loader (torch.utils.data.dataloader.DataLoader): train data loader.
            optimizer (torch.optim): optimizer

        Returns: loss (float)
    """
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    return loss.item()


def test(model, device, test_loader) -> Tuple[float, float, torch.Tensor]:
    """Test function

        Arguments:
            model (nn.Module): some networks extends from ``nn.Module``.
            device (torch.device): device for use CUDA if available.
            test_loader (torch.utils.data.dataloader.DataLoader): test data loader.

        Returns: tuple of below three
            test_loss:
            accuracy:
            output:
    """
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy, output


torch.manual_seed(42)  # 42, THE ANSWER TO LIFE, THE UNIVERSE AND EVERYTHING

batch = 64  # batch size
lr = .01  # learning rate
epochs = 10

TRAIN_DATASET.transform = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(TRAIN_DATASET,
                                           batch_size=64,
                                           shuffle=True)

TEST_DATASET.transform = transforms.ToTensor()
test_loader = torch.utils.data.DataLoader(TEST_DATASET,
                                          batch_size=64,
                                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Network().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    train_loss = train(model, device, train_loader, optimizer)
    test_loss, accuracy, _ = test(model, device, test_loader)

    print('Epoch: {}\t Loss: {:.6f}'.format(epoch, train_loss))
    print('\t\t Average Loss: {:.4f}, Accuracy: {:.0f}%'.format(test_loss, accuracy))

torch.save(model.state_dict(), "mnist_cnn.pt")  # save trained model named 'mnist_cnn.pt'
