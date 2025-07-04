import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioMLP(nn.Module):
    def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_reduce = time_reduce
        # optimised for GPU, faster than x.reshape(*x.shape[:-1], -1, 2).mean(-1)
        self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce)  # Non-overlapping averaging

        self.fc1 = nn.Linear(n_steps * n_mels, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # reduce time dimension
        shape = x.shape
        x = x.reshape(-1, 1, x.shape[-1])
        x = self.pool(x)  # (4096, 1, 431)
        x = x.reshape(shape[0], shape[1], shape[2], -1)

        # 2D to 1D
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    """
    A minimal residual block without downsampling
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # skip connection
        out = self.relu(out)
        return out

# Simple ResNet
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.res_block1 = ResidualBlock(16)
        self.res_block2 = ResidualBlock(16)
        self.res_block3 = ResidualBlock(16)
        self.res_block4 = ResidualBlock(16)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


