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
from tqdm import tqdm
from collections import defaultdict
import time

def get_model(model_name, dropout_rate, initializer='xavier'):
    if model_name == 'resnet':
        model = resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 6)
        )
    elif model_name == 'vgg16':
        model = vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 6)
        )
    elif model_name == 'vit':
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        num_ftrs = model.config.hidden_size
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 6)
        )
    else:
        raise ValueError("Invalid model name. Choose from 'resnet', 'vgg16', 'vit'.")
    if initializer == 'xavier':
        nn.init.xavier_uniform_(model.fc.weight)
    elif initializer == 'he':
        nn.init.kaiming_uniform_(model.fc.weight)
    elif initializer == 'normal':
        nn.init.normal_(model.fc.weight)
    # you can add more initializers here

    return model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help="Model name: 'resnet', 'vgg16', or 'vit'")
parser.add_argument('--epochs', type=int, default = 20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
args = parser.parse_args()

def train_model(model, criterion, optimizer, train_loader, epochs, device):
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}, Training")
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            pbar.update()
        pbar.close()

        train_losses.append(train_loss / len(train_loader.dataset))

    return train_losses


def test_model(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    test_losses = []
    pbar = tqdm(total=len(test_loader), desc=f"Testing")

    avg_inference_time = 0
    count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            start = time.time()

            outputs = model(inputs)
            
            end = time.time() 
            avg_inference_time += end - start
            count += 1

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.update()
    pbar.close()

    avg_inference_time /= count
    test_times.append(avg_inference_time)

    test_losses.append(test_loss / len(test_loader.dataset))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch_metrics = (accuracy, precision, recall)
        best_conf_matrix = conf_matrix

avg_time_all_epochs = sum(test_times) / len(test_times)
with open(f'{args.model}_inference_times.txt', 'w') as f:
    f.write("Average Inference Time of Batch, each Epoch")
    for t in range(len(test_times)):
        f.write(f"Epoch {t + 1}: {test_times[t]}")
    f.write(f"Average Inference Time of Batch, all Epochs: {avg_time_all_epochs}")

print('Training and testing complete.')

# Save model, metrics, and confusion matrix from the best epoch
torch.save(best_model.state_dict(), f'{args.model}_best_model.pth')

with open(f'results/{args.model}_best_metrics.txt', 'w') as f:
    f.write(f'Best metrics:\n')
    f.write(f'Accuracy: {best_metrics["accuracy"]}\n')
    f.write(f'Precision: {best_metrics["precision"]}\n')
    f.write(f'Recall: {best_metrics["recall"]}\n')

np.save(f'results/{args.model}_best_confusion_matrix.npy', best_metrics['conf_matrix'])
sns.heatmap(best_metrics['conf_matrix'], annot=True)
plt.savefig(f'results/{args.model}_best_confusion_matrix.png')

# Plot performance curve of best model
plt.figure()
plt.plot(range(args.epochs), train_losses, label='Train Loss')
plt.plot(range(args.epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'results/{args.model}_performance_curve.png')
