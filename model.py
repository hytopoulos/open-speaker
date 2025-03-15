import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torchvision.models import resnet34, resnet50, resnet101

class SpeakerNetTransformer(nn.Module):
    def __init__(self, config):
        super(SpeakerNetTransformer, self).__init__()
        self.transformer = TransformerEncoder(
            nn.TransformerEncoderLayer(128, 8, dim_feedforward=256, batch_first=True),
            num_layers=6,
        )

        self.dropout = nn.Dropout(config.params.dropout)
        self.fc = nn.Linear(128, 128)

    def forward(self, x, masks):
        x = x.transpose(1, 2) # swap axes to (batch_size, time_dim, feature_dim)
        x = self.transformer(x, src_key_padding_mask=masks)
        # pool over time dimension
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class SpeakerResNet34(nn.Module):
    def __init__(self, config):
        super(SpeakerResNet34, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet34(num_classes=config.params.resnet_dim)

    def forward(self, x, _):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.resnet(x)
        x = x.squeeze(1)
        return x

class SpeakerResNet50(nn.Module):
    def __init__(self, config):
        super(SpeakerResNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet50(num_classes=config.params.resnet_dim)

    def forward(self, x, _):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.resnet(x)
        x = x.squeeze(1)
        return x
    
class SpeakerResNet101(nn.Module):
    def __init__(self, config):
        super(SpeakerResNet101, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=config.params.init_k)
        self.resnet = resnet101(num_classes=config.params.resnet_dim)

    def forward(self, x, _):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.resnet(x)
        x = x.squeeze(1)
        return x