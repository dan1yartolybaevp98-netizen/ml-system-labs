import torch
from config import CHECKPOINT_DIR, BASELINE_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, ensure_dirs, get_device
from dataset import get_loaders
from models import BaselineCNN, count_parameters
from utils import set_seed, train_epochs, evaluate, measure_latency, model_size_mb, save_json

def run_baseline(epochs=BASELINE_EPOCHS):
    ensure_dirs()
    set_seed()
    device = get_device()
    print(f"Device: {device}")
    train_loader, test_loader = get_loaders()
    model = BaselineCNN().to(device)
    params = count_parameters(model)
    print(f"Baseline parameters: {params:,}")
    history = train_epochs(
        model, train_loader, test_loader, epochs, LEARNING_RATE,
        WEIGHT_DECAY, device, label="Baseline"
    )
    _, accuracy = evaluate(model, test_loader, device)
    checkpoint = CHECKPOINT_DIR / "baseline_cnn.pt"
    torch.save(model.cpu().state_dict(), checkpoint)
    model.to(device)
    latency = measure_latency(model, device)
    metrics = {
        "model": "Baseline CNN", "parameters": params, "accuracy_percent": accuracy,
        "checkpoint_size_mb": model_size_mb(checkpoint),
        "latency_ms_per_image_batch_1": latency, "device": str(device),
        "epochs": epochs, "history": history
    }
    save_json(metrics, CHECKPOINT_DIR / "baseline_metrics.json")
    print("Baseline metrics:", metrics)
    return metrics

if __name__ == "__main__":
    run_baseline()
