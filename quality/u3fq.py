import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNetWithFeatures(nn.Module):
    def __init__(self, num_feature_vector=3):
        super(ResNetWithFeatures, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048 + num_feature_vector, 512)  # Concatenate resnet output with feature vector
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, img, features):
        x1 = self.resnet(img)
        x1 = self.flatten(x1)
        features = features.unsqueeze(1) if features.dim() == 1 else features
        x = torch.cat((x1, features), dim=1)  # Concatenate along the feature dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    