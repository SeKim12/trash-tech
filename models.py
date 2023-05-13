import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, vgg16
from transformers import ViTModel
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data.pipeline import generate_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_model(model_name):
    if model_name == 'resnet':
        model = resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 6)
    elif model_name == 'vgg16':
        model = vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 6)
    elif model_name == 'vit':
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        num_ftrs = model.config.hidden_size
        model.classifier = nn.Linear(num_ftrs, 6)
    else:
        raise ValueError("Invalid model name. Choose from 'resnet', 'vgg16', 'vit'.")
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Model name: 'resnet', 'vgg16', or 'vit'")
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
args = parser.parse_args()

model = get_model(args.model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataset = generate_split()
train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

all_labels = []
all_preds = []
for epoch in range(args.epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

# Compute metrics after all epochs are completed
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Save final model and metrics
torch.save(model.state_dict(), 'final_model.pth')

with open('final_metrics.txt', 'w') as f:
    f.write(f'Final Metrics after {args.epochs} epochs:\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')

# Save and visualize final confusion matrix
np.save('final_confusion_matrix.npy', conf_matrix)
sns.heatmap(conf_matrix, annot=True)
plt.savefig('final_confusion_matrix.jpg')