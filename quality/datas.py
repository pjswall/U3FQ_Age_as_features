from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd 
import os

class FaceQualityDataset(Dataset):
    def __init__(self, csv_path, img_folder, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image']
        img_path = os.path.join(self.img_folder, img_name)  
        age = self.data.iloc[idx]['Image1_Age'] / 100.0  
        expression = self.data.iloc[idx]['Emotion_Similarity'] 
        avg_match_score = self.data.iloc[idx]['Avg_N_Score']
        label = self.data.iloc[idx]['label']
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        features = torch.tensor([age, expression, avg_match_score], dtype=torch.float32)
        
        return img, features, label
