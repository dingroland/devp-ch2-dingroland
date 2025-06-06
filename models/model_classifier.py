import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AudioMLP(nn.Module):
    def __init__(self, n_mels, n_steps, hidden1_size, hidden2_size, output_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(n_steps * n_mels, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 2D to 1D
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, n_mels=128, n_steps=431):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # Correctly calculate the flattened size
        conv_output_height = n_mels // 2 // 2
        conv_output_width = (n_steps // 2 // 2) 
        self.fc1 = nn.Linear(64 * conv_output_height * conv_output_width, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNetForAudio(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        # Modify the first convolutional layer to accept 1-channel (grayscale) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer for the number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        return self.resnet(x)
