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


# better ResNet

class BetterResidualBlock(nn.Module):
    """
    Basic residual block: Conv2D -> BN -> ReLU -> Conv2D -> BN + skip connection.
    Optionally downsamples the input via stride=2 and projection conv.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1

        # First conv with possible stride for downsampling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv keeps output shape
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Projection layer for skip connection if shape changes
        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    - initial conv layer
    - 4 residual stages with increasing channels and optional downsampling
    - global average pooling
    - dropout
    - final linear classification head
    """
    def __init__(self, num_classes=50):
        super().__init__()
        self.in_channels = 16

        # Initial conv (input is 1-channel Mel-spectrogram)
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages (output channels increase, downsampling)
        self.layer1 = BetterResidualBlock(16, 32, downsample=True)
        self.layer2 = BetterResidualBlock(32, 64, downsample=True)
        self.layer3 = BetterResidualBlock(64, 128, downsample=True)
        self.layer4 = BetterResidualBlock(128, 128)
        self.layer5 = BetterResidualBlock(128, 128)


        # Global average pooling to reduce to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))  # Shape: (B, 16, H, W)

        x = self.layer1(x)  # -> (B, 32, H/2, W/2)
        x = self.layer2(x)  # -> (B, 64, H/4, W/4)
        x = self.layer3(x)  # -> (B, 128, H/8, W/8)
        x = self.layer4(x)  # -> (B, 128, H/8, W/8)
        x = self.layer5(x)  # -> (B, 128, H/8, W/8)


        x = self.global_pool(x)  # -> (B, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 128)

        x = self.dropout(x)
        return self.fc(x)  # Final shape: (B, num_classes)