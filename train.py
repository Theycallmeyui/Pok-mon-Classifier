import os
import copy
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score


DATA_DIR = "data"
BATCH_SIZE = 16
EPOCHS = 5
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


experiments = [
    {
        "name": "resnet18_pretrained_freeze",
        "model": "resnet18",
        "pretrained": True,
        "freeze": True,
    },
    {
        "name": "resnet18_pretrained_finetune",
        "model": "resnet18",
        "pretrained": True,
        "freeze": False,
    },
    {
        "name": "mobilenet_pretrained_freeze",
        "model": "mobilenet",
        "pretrained": True,
        "freeze": True,
    },
    {
        "name": "resnet18_no_pretrained",
        "model": "resnet18",
        "pretrained": False,
        "freeze": False,
    },
]


def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tf)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, train_data.classes


def build_model(model_name, pretrained, freeze, num_classes):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Unknown model name")

    return model.to(DEVICE)


def train_one_experiment(exp, train_loader, test_loader, num_classes):
    print(f"\n===== Experiment: {exp['name']} =====")

    model = build_model(exp["model"], exp["pretrained"], exp["freeze"], num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    train_losses = []
    test_accuracies = []

    best_model = None
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        acc, precision, recall = evaluate(model, test_loader)
        test_accuracies.append(acc)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model.state_dict())

    os.makedirs("models", exist_ok=True)
    torch.save(best_model, f"models/{exp['name']}.pth")

    plot_learning_curve(exp["name"], train_losses, test_accuracies)

    acc, precision, recall = evaluate(model, test_loader)

    return {
        "experiment": exp["name"],
        "model": exp["model"],
        "pretrained": exp["pretrained"],
        "freeze": exp["freeze"],
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
    }


def evaluate(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, precision, recall


def plot_learning_curve(name, losses, accuracies):
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(losses, label="Train Loss")
    plt.plot(accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(name)
    plt.legend()
    plt.savefig(f"results/{name}_learning_curve.png")
    plt.close()


def main():
    train_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    print("Number of classes:", num_classes)
    print("Device:", DEVICE)

    os.makedirs("models", exist_ok=True)
    with open("models/classes.json", "w") as f:
        json.dump(class_names, f)

    all_results = []

    for exp in experiments:
        result = train_one_experiment(exp, train_loader, test_loader, num_classes)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/experiment_results.csv", index=False)

    print("\nFinal Results:")
    print(df)


if __name__ == "__main__":
    main()