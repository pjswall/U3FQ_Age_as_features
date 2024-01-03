import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import ParameterSampler
import pandas as pd
from tqdm import tqdm
from u3fq import FaceQualityDataset, ResNetWithFeatures
import numpy as np


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Assuming FaceQualityDataset and ResNetWithFeatures are already defined as shown previously

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for img, features, label in train_loader:
        img, features, label = img.to(device), features.to(device), label.to(device)

        # Forward pass
        outputs = model(img, features)
        loss = criterion(outputs, label.unsqueeze(1).float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def hyperparameter_tuning(img_folder, csv_path, param_grid, n_iter, num_epochs, device):
    # Prepare the dataset and transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Randomly sample hyperparameters
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter))

    best_loss = float('inf')
    best_params = {}

    for params in param_list:
        print(f"Training with parameters: {params}")

        # Initialize model, criterion, optimizer, and dataloader with current set of hyperparameters
        model = ResNetWithFeatures().to(device)
        criterion = nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=params['lr'])
        train_loader = DataLoader(
            FaceQualityDataset(csv_path=csv_path,img_folder = img_folder , transform=transform),
            batch_size=params['batch_size'], 
            shuffle=True
        )

        # Train the model for one epoch
        epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Check if this is the best hyperparameters so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_params = params
            torch.save(model.state_dict(), 'best_model_weights.pth')  # Save the best model weights

    return best_params, best_loss

# Define the parameter grid
param_grid = {
    'lr': [1e-3, 1e-4, 1e-5],
    'batch_size': [8, 16, 32, 64]
}

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter tuning
best_params, best_loss = hyperparameter_tuning(
    img_folder = '../AgeDB/',
    csv_path='./train.csv',
    param_grid=param_grid,
    n_iter=1,
    num_epochs=1,  # For hyperparameter tuning, you might want to train for fewer epochs
    device=device
)

print(f"Best hyperparameters: {best_params} with loss: {best_loss}")



#scp -o 'ProxyJump=rktp27@10.2.204.204' -r ../../Quality_Metric/Face/AEM_scores.csv prateek.j@biosid:~/FIQ/