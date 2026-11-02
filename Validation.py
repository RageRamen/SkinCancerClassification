import torch
import numpy as np
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os

metadata = pd.read_csv("Dataset/HAM10000_metadata.csv")

folder1 = "Dataset/HAM10000_images_part_1"
folder2 = "Dataset/HAM10000_images_part_2"

image_path_dict = {}
for index, row in metadata.iterrows():
    img_id = row['image_id']

    if os.path.exists(os.path.join(folder1, img_id + '.jpg')):
        image_path_dict[img_id] = os.path.join(folder1, img_id + '.jpg')
    elif os.path.exists(os.path.join(folder2, img_id + '.jpg')):
        image_path_dict[img_id] = os.path.join(folder2, img_id + '.jpg')
    #else:
    #    raise ValueError(f"Image {img_id} not found in either folder")

metadata['path'] = metadata['image_id'].map(image_path_dict)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
metadata['label'] = le.fit_transform(metadata['dx'])

train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])

class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.df.iloc[index]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = SkinLesionDataset(test_df, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU available: {torch.cuda.is_available()}")
print(device)

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.to(device)

model.load_state_dict(torch.load("models/resnet50_v1/resnet50_v1_epoch_10.pth"))
model.eval()

def evaluate_model(model, loader, device, classes):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
evaluate_model(model, val_loader, device, classes)

