import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total

def model_size_mb(path):
    return Path(path).stat().st_size / (1024 ** 2)

def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def measure_latency(model, device, input_shape=(1, 1, 28, 28), warmup=30, iterations=150):
    model.eval()
    sample = torch.randn(*input_shape, device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000.0

def train_epochs(model, train_loader, test_loader, epochs, lr, weight_decay, device, label="Training"):
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
            "test_loss": test_loss, "test_accuracy": test_acc
        }
        history.append(row)
        print(f"{label} | epoch {epoch}/{epochs} | "
              f"train acc {train_acc:.2f}% | test acc {test_acc:.2f}%")
    return history
