import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow
import os

from networkx import dfs_successors

metadata = pd.read_csv("Dataset/HAM10000_metadata.csv")
print(metadata.head())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

found_count = metadata['path'].notna().sum()
print(f"Found {found_count} images out of {len(metadata)}")
print(metadata[['image_id', 'path']].head())

#img = mpimg.imread(metadata['path'][0])
#plt.imshow(img)
#plt.show()

# Checks distribution of skin lesion types
#print(metadata['dx'].value_counts())
#metadata['dx'].value_counts().plot(kind='bar')
#plt.show()

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = cv2.resize(image, (224, 224))
    return image / 255.0

sample_img = load_and_preprocess_image(metadata['path'][0])
print(f"New shape: {sample_img.shape}")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
metadata['label'] = le.fit_transform(metadata['dx'])

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['label'])

train_dataset = SkinLesionDataset(train_df, transform=data_transforms)
val_dataset = SkinLesionDataset(test_df, transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Tran size: {len(train_df)}, Test size: {len(test_df)}")

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(metadata['label'])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=metadata['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float32)

print(class_weights)

from torchvision import models

model = models.resnet50(weights='DEFAULT')

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU available: {torch.cuda.is_available()}")
print(device)
class_weights = class_weights.to(device)
model.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import os
from tqdm import tqdm

output_dir = "models/resnet50_v1"
os.makedirs(output_dir, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for data in pbar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=running_loss / len(loader), acc=100.*correct / total)

num_epochs = 10
best_acc = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_one_epoch(model, train_loader, optimizer, criterion, device)

    torch.save(model.state_dict(), os.path.join(output_dir, f"resnet50_v1_epoch_{epoch + 1}.pth"))
