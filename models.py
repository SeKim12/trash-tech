import torch
import torch.nn as nn
from torchvision.models import resnet50, vgg16, vit_b_16
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from tqdm import tqdm
import time
import os


def get_model(model_name, dropout_rate, initializer="xavier"):
    init = None
    if initializer == "xavier":
        init = nn.init.xavier_uniform_
    elif initializer == "he":
        init = nn.init.kaiming_uniform_
    else:
        init = nn.init.normal_

    if model_name == "resnet":
        print("Retreiving Pre-trained ResNet50...")
        model = resnet50(weights="ResNet50_Weights.DEFAULT")
        num_ftrs = model.fc.in_features

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(num_ftrs, 6))

        init(model.fc[1].weight)

    elif model_name == "vgg16":
        print("Retreiving VGG16...")
        model = vgg16(weights="VGG16_Weights.DEFAULT")
        num_ftrs = model.classifier[6].in_features

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(num_ftrs, 6)
        )

        init(model.classifier[6][1].weight)

    elif model_name == "vit":
        # model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        print("Retreiving ViT...")
        model = vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
        num_ftrs = model.heads[0].in_features

        for param in model.parameters():
            param.requires_grad = False

        model.heads[0] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(num_ftrs, 6))

        init(model.heads[0][1].weight)

    else:
        raise ValueError("Invalid model name. Choose from 'resnet', 'vgg16', 'vit'.")

    return model


def train_model(
    model,
    config,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs,
    device,
    glob_best_val_acc,
):
    train_losses = []
    val_losses = []

    # save best metrics for each hyperparameter configuration
    # i.e. record all metrics, but save only the best (across all hyperparameters) model
    best_val_metrics = {}

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

        # validate per epoch
        val_metrics, val_loss, _ = test_model(model, criterion, val_loader, device)

        # save best validation metric (for each hyperparameter)
        if val_metrics["accuracy"] > best_val_metrics.get("accuracy", -1):
            best_val_metrics = val_metrics

        # only save best model (across all hyperparameters) for testing
        if val_metrics["accuracy"] > glob_best_val_acc:
            print(
                "New Best Validation Accuracy: {:.4f}".format(val_metrics["accuracy"])
            )

            glob_best_val_acc = val_metrics["accuracy"]
            to_save = model.module if hasattr(model, "module") else model
            torch.save(
                dict(
                    epoch=epoch,
                    model_state_dict=to_save.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    args=config,
                ),
                os.path.join(
                    config["checkpoint_dir"], "{}.pt".format(config["model_name"])
                ),
            )

        val_losses.append(val_loss)

    return train_losses, val_losses, val_metrics, glob_best_val_acc


def test_model(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
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

    avg_inference_time /= count  # this is average time / batch
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    test_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "test_loss": test_loss,
        "inference_time_per_batch": avg_inference_time,
        "total_inference_time": avg_inference_time * count,
    }

    return test_metrics, test_loss, conf_matrix
