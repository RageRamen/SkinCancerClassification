import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU available: {torch.cuda.is_available()}")
print(device)

model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.to(device)

model.load_state_dict(torch.load("models/resnet50_v1/resnet50_v1_epoch_10.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    return classes[predicted.item()]

print(predict_image("Dataset/HAM10000_images_part_1/ISIC_0024350.jpg"))

