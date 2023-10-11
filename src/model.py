import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
import torch.nn as nn

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)  
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        # Define linear layers (MLP)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 1024)
        self.bn5 = nn.BatchNorm1d(1024)  
        self.dp1 = nn.Dropout(p=dropout)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu1(self.pool1(self.bn1(self.conv1(x))))
        x = self.relu2(self.pool2(self.bn2(self.conv2(x))))
        x = self.relu3(self.pool3(self.bn3(self.conv3(x))))
        x = self.relu4(self.pool4(self.bn4(self.conv4(x))))

        x = self.flatten(x)
        x = self.rl1(self.dp1(self.bn5(self.fc1(x))))
        x = self.fc2(x)

        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
