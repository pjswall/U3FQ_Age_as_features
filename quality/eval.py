import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from u3fq import ResNetWithFeatures
from datas import FaceQualityDataset

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predicted_labels = []
    all_actual_labels = []
    results = []

    with torch.no_grad():
        for batch_idx, (img, features, labels) in enumerate(tqdm(data_loader)):
            img, features = img.to(device), features.to(device)
            outputs = model(img, features).squeeze()
            predicted_scores = outputs.cpu().numpy()
            predicted_labels = torch.round(outputs).cpu().numpy()
            actual_labels = labels.numpy()

            all_predicted_labels.extend(predicted_labels)
            all_actual_labels.extend(actual_labels)

            start_idx = batch_idx * data_loader.batch_size
            end_idx = start_idx + len(labels)
            for idx, label_idx in enumerate(range(start_idx, end_idx)):
                img_name = data_loader.dataset.data.iloc[label_idx]['Image']
                results.append((img_name, predicted_scores[idx]))

    accuracy = accuracy_score(all_actual_labels, all_predicted_labels)
    precision = precision_score(all_actual_labels, all_predicted_labels)
    recall = recall_score(all_actual_labels, all_predicted_labels)
    f1 = f1_score(all_actual_labels, all_predicted_labels)

    return accuracy, precision, recall, f1, results

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader for evaluation
val_csv_path = './quality/test.csv'
img_folder = './AgeDB/'  # Set your image folder path here
val_dataset = FaceQualityDataset(csv_path=val_csv_path, img_folder=img_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetWithFeatures().to(device)
model.load_state_dict(torch.load('./quality/model_weights.pth'))  # Load your trained model weights

# Evaluate the model
accuracy, precision, recall, f1, results = evaluate_model(model, val_loader, device)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

results_csv_path = './quality/quality_scores.csv'
with open(results_csv_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Quality_Score'])
    for result in results:
        writer.writerow(result)

print(f"Quality scores saved to {results_csv_path}")
