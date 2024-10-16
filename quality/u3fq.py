import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from arcface import ArcFaceModel  # Replace with the actual module name for ArcFace

class CustomFaceModel(nn.Module):
    def __init__(self, num_feature_vector=3, num_classes=len(np.load("path/to/names.npy"))):
        super(CustomFaceModel, self).__init__()

        # Load pre-trained ArcFace model
        self.arcface = ArcFaceModel(pretrained=True)

        # Remove the last layer of the ArcFace model
        self.arcface = nn.Sequential(*list(self.arcface.children())[:-1])

        # Add your custom layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 + num_feature_vector, 512)  # Adjust input size based on the ArcFace model
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)  # Output layer with the number of classes
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, features):
        x1 = self.arcface(img)
        x1 = self.flatten(x1)
        features = features.unsqueeze(1) if features.dim() == 1 else features
        x = torch.cat((x1, features), dim=1)  # Concatenate along the feature dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

# Example of how to load the ArcFace model and names
arcface_model = ArcFaceModel(pretrained=True)
names = np.load("path/to/names.npy")

# Example of creating an instance of your custom model
custom_model = CustomFaceModel()

# Training loop:
# - Load images and labels from the organized facebank-like dataset
# - Use a suitable loss function (e.g., nn.BCEWithLogitsLoss) for binary classification
# - Fine-tune the model with your training data
# - Save the trained model for later use


# import torch
# import torch.nn as nn
# from torchvision.models import resnet50


# class ResNetWithFeatures(nn.Module):
#     def __init__(self, num_feature_vector=3):
#         super(ResNetWithFeatures, self).__init__()
#         self.resnet = resnet50(pretrained=True)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(2048 + num_feature_vector, 512)  # Concatenate resnet output with feature vector
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(512, 1)
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, img, features):
#         x1 = self.resnet(img)
#         x1 = self.flatten(x1)
#         features = features.unsqueeze(1) if features.dim() == 1 else features
#         x = torch.cat((x1, features), dim=1)  # Concatenate along the feature dimension
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
        
#         return x
    