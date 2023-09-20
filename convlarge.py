#!coding:utf-8
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class CNN(nn.Module):

    def __init__(self, num_classes, drop_ratio=0.0, code_bits=12, pretrained=True):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.drop_ratio = drop_ratio
        self.code_bits = code_bits
        original_model = models.alexnet(pretrained)
        self.features = original_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        fc1 = nn.Linear(256 * 6 * 6, 4096)  # 256/512
        fc2 = nn.Linear(4096, 4096)
        if pretrained:
            fc1.weight = original_model.classifier[1].weight
            fc1.bias = original_model.classifier[1].bias
            fc2.weight = original_model.classifier[4].weight
            fc2.bias = original_model.classifier[4].bias

        self.backbone = nn.Sequential(
            nn.Dropout(self.drop_ratio),
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),
            fc2,
            nn.ReLU(inplace=True)
        )

        self.hash_layer = nn.Sequential(
            nn.Linear(4096, self.code_bits),
            nn.Tanh()
        )

        self.classifier_class = nn.Sequential(
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        feature = self.features(x)
        feature = self.avgpool(feature)
        out = feature.view(feature.size(0), 256 * 6 * 6)
        fc7 = self.backbone(out)
        class_code = self.classifier_class(fc7)
        hash_code = self.hash_layer(fc7)
        return fc7, class_code, hash_code

