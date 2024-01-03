import torch
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from u3fq import ResNetWithFeatures
from datas import FaceQualityDataset

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader for training
img_folder = '/home/prateek.j/FIQ/AgeDB/'
train_csv_path = '/home/prateek.j/FIQ/quality/train.csv'
train_dataset = FaceQualityDataset(csv_path=train_csv_path, img_folder=img_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetWithFeatures().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    for img, features, label in tqdm(train_loader):
        img, features, label = img.to(device), features.to(device), label.to(device).float()

        # Forward pass
        outputs = model(img, features).squeeze()

        # Compute loss
        loss = criterion(outputs, label.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Save the trained model weights
torch.save(model.state_dict(), '/home/prateek.j/FIQ/quality/model_weights.pth')
