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
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.update()
    pbar.close()

    test_losses.append(test_loss / len(test_loader.dataset))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'conf_matrix': conf_matrix,
        'test_losses': test_losses,
    }

    return test_metrics, test_losses

learning_rates = [0.1, 0.01, 0.001]
dropout_rates = [0.1, 0.2, 0.3]
initializers = ['xavier', 'he', 'normal']

dataset = generate_split()
train_loader = DataLoader(dataset['train'], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_accuracy = 0
best_metrics = defaultdict(list)
best_hyperparams = None
best_model = None
best_train_losses = None
best_test_losses = None

# perform a grid search over all combinations of hyperparameters
for lr in learning_rates:
    for dropout_rate in dropout_rates:
        for initializer in initializers:
            model = get_model('resnet', dropout_rate, initializer)  # replace 'resnet' with your preferred model
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_losses = train_model(model, criterion, optimizer, train_loader, args.epochs)
            test_metrics, test_losses = test_model(model, criterion, test_loader)

            # if the model with the current set of hyperparameters
            # has a better accuracy than the previous best model,
            # update the best accuracy and best set of hyperparameters
            if test_metrics['accuracy'] > best_accuracy:
                best_model = model
                best_metrics = test_metrics
                best_hyperparams = {'learning_rate': lr, 'dropout_rate': dropout_rate, 'initializer': initializer}
                best_train_losses = train_losses
                best_test_losses = test_losses


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
