import copy
import csv
import torch
import torch.nn.utils.prune as prune
from config import CHECKPOINT_DIR, RESULTS_DIR, PRUNING_FINE_TUNE_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, ensure_dirs, get_device
from dataset import get_loaders
from models import BaselineCNN, count_parameters
from utils import set_seed, train_epochs, evaluate, measure_latency, model_size_mb

SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
STRUCTURED_LEVELS = [0.10, 0.25, 0.50]

def eligible_modules(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]

def apply_unstructured(model, amount, mode):
    pairs = eligible_modules(model)
    if amount <= 0:
        return
    if mode == "global":
        tensors = [module.weight.detach().abs().flatten() for _, module in pairs]
        all_scores = torch.cat(tensors)
        count = int(round(amount * all_scores.numel()))
        if count == 0:
            return
        threshold_indices = torch.argsort(all_scores)[:count]
        global_mask = torch.ones_like(all_scores, dtype=torch.bool)
        global_mask[threshold_indices] = False
        offset = 0
        for _, module in pairs:
            n = module.weight.numel()
            mask = global_mask[offset:offset+n].reshape_as(module.weight).to(module.weight.device)
            prune.custom_from_mask(module, name="weight", mask=mask)
            offset += n
    elif mode == "layerwise":
        for _, module in pairs:
            prune.l1_unstructured(module, name="weight", amount=amount)
    else:
        raise ValueError("mode must be 'global' or 'layerwise'")

def apply_structured(model, amount):
    # Prune whole Conv2d output filters by L2 norm and also zero the matching
    # BatchNorm channel so a removed filter stays zero after normalization.
    modules = list(model.features.children())
    for index, module in enumerate(modules):
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            mask = module.weight_mask.detach().reshape(module.out_channels, -1).sum(dim=1) > 0
            if index + 1 < len(modules) and isinstance(modules[index + 1], torch.nn.BatchNorm2d):
                bn = modules[index + 1]
                channel_mask = mask.to(bn.weight.device, dtype=bn.weight.dtype)
                if bn.affine:
                    prune.custom_from_mask(bn, name="weight", mask=channel_mask)
                    prune.custom_from_mask(bn, name="bias", mask=channel_mask)

def finalize_pruning(model):
    # Remove pruning re-parameterizations after masked fine-tuning; zeros remain.
    for module in model.modules():
        for parameter_name in ("weight", "bias"):
            if hasattr(module, parameter_name + "_orig"):
                prune.remove(module, parameter_name)

def actual_weight_sparsity(model):
    total, zeros = 0, 0
    for _, module in eligible_modules(model):
        w = module.weight.detach()
        total += w.numel()
        zeros += (w == 0).sum().item()
    return zeros / total if total else 0.0

def load_teacher(device):
    model = BaselineCNN().to(device)
    path = CHECKPOINT_DIR / "baseline_cnn.pt"
    if not path.exists():
        raise FileNotFoundError("Run baseline.py first to create baseline_cnn.pt")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model

def run_pruning():
    ensure_dirs()
    set_seed()
    device = get_device()
    train_loader, test_loader = get_loaders()
    baseline = load_teacher(device)
    rows = []

    for mode in ["global", "layerwise"]:
        for amount in SPARSITY_LEVELS:
            print(f"\\n{mode} pruning target={amount:.0%}")
            model = copy.deepcopy(baseline)
            apply_unstructured(model, amount, mode)
            if amount > 0:
                train_epochs(model, train_loader, test_loader, PRUNING_FINE_TUNE_EPOCHS,
                             LEARNING_RATE * 0.3, WEIGHT_DECAY, device,
                             label=f"{mode} {amount:.0%}")
            sparsity = actual_weight_sparsity(model)
            _, accuracy = evaluate(model, test_loader, device)
            finalize_pruning(model)
            path = CHECKPOINT_DIR / f"{mode}_pruning_{int(amount*100)}.pt"
            torch.save(model.cpu().state_dict(), path)
            model.to(device)
            latency = measure_latency(model, device)
            rows.append({
                "method": mode, "target_sparsity": amount, "actual_sparsity": sparsity,
                "accuracy_percent": accuracy, "parameters": count_parameters(model),
                "size_mb": model_size_mb(path), "latency_ms_per_image_batch_1": latency,
                "device": str(device)
            })

    for amount in STRUCTURED_LEVELS:
        print(f"\\nstructured pruning target={amount:.0%}")
        model = copy.deepcopy(baseline)
        apply_structured(model, amount)
        train_epochs(model, train_loader, test_loader, PRUNING_FINE_TUNE_EPOCHS,
                     LEARNING_RATE * 0.3, WEIGHT_DECAY, device,
                     label=f"structured {amount:.0%}")
        sparsity = actual_weight_sparsity(model)
        _, accuracy = evaluate(model, test_loader, device)
        finalize_pruning(model)
        path = CHECKPOINT_DIR / f"structured_pruning_{int(amount*100)}.pt"
        torch.save(model.cpu().state_dict(), path)
        model.to(device)
        latency = measure_latency(model, device)
        rows.append({
            "method": "structured", "target_sparsity": amount, "actual_sparsity": sparsity,
            "accuracy_percent": accuracy, "parameters": count_parameters(model),
            "size_mb": model_size_mb(path), "latency_ms_per_image_batch_1": latency,
            "device": str(device)
        })

    csv_path = RESULTS_DIR / "pruning_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved pruning results to {csv_path}")
    return rows

if __name__ == "__main__":
    run_pruning()
