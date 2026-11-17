import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from config import (CHECKPOINT_DIR, RESULTS_DIR, STUDENT_EPOCHS, LEARNING_RATE,
                    WEIGHT_DECAY, TEMPERATURE, ALPHA, ensure_dirs, get_device)
from dataset import get_loaders
from models import BaselineCNN, StudentCNN, count_parameters
from utils import set_seed, evaluate, measure_latency, model_size_mb

def load_teacher(device):
    teacher = BaselineCNN().to(device)
    path = CHECKPOINT_DIR / "baseline_cnn.pt"
    if not path.exists():
        raise FileNotFoundError("Run baseline.py first to create baseline_cnn.pt")
    teacher.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher

def train_student(student, train_loader, test_loader, device, use_distillation,
                  epochs=STUDENT_EPOCHS):
    teacher = load_teacher(device) if use_distillation else None
    optimizer = optim.AdamW(student.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    ce_loss = nn.CrossEntropyLoss()
    history = []
    student.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)
            loss_ce = ce_loss(student_logits, labels)
            if use_distillation:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                # Implements KL(student/T || teacher/T), matching the lab sheet's order.
                student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=1)
                teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=1)
                loss_kl = F.kl_div(
                    F.log_softmax(teacher_logits / TEMPERATURE, dim=1),
                    F.softmax(student_logits / TEMPERATURE, dim=1),
                    reduction="batchmean"
                )
                loss = ALPHA * loss_ce + (1.0 - ALPHA) * (TEMPERATURE ** 2) * loss_kl
            else:
                loss = loss_ce
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (student_logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        _, test_acc = evaluate(student, test_loader, device)
        row = {"epoch": epoch, "train_loss": running_loss / total,
               "train_accuracy": 100.0 * correct / total, "test_accuracy": test_acc}
        history.append(row)
        print(f"{'KD Student' if use_distillation else 'Plain Student'} "
              f"| epoch {epoch}/{epochs} | test acc {test_acc:.2f}%")
    return history

def run_distillation():
    ensure_dirs()
    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders()
    rows = []
    for use_kd, label in [(False, "student_plain"), (True, "student_distilled")]:
        student = StudentCNN().to(device)
        params = count_parameters(student)
        print(f"{label} parameters: {params:,}")
        history = train_student(student, train_loader, test_loader, device, use_kd)
        _, accuracy = evaluate(student, test_loader, device)
        path = CHECKPOINT_DIR / f"{label}.pt"
        torch.save(student.cpu().state_dict(), path)
        student.to(device)
        latency = measure_latency(student, device)
        rows.append({
            "model": label, "parameters": params, "accuracy_percent": accuracy,
            "size_mb": model_size_mb(path),
            "latency_ms_per_image_batch_1": latency, "device": str(device),
            "temperature": TEMPERATURE if use_kd else None,
            "alpha": ALPHA if use_kd else None
        })
    csv_path = RESULTS_DIR / "distillation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved distillation results to {csv_path}")
    return rows

if __name__ == "__main__":
    run_distillation()
