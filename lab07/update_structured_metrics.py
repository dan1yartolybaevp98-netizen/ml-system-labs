import csv
import torch
from config import CHECKPOINT_DIR, RESULTS_DIR
from models import BaselineCNN

csv_path = RESULTS_DIR / "pruning_results.csv"

def count_filter_sparsity(model):
    total = 0
    zero = 0

    for module in model.features.modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.detach()
            filters = weights.reshape(module.out_channels, -1)
            norms = torch.linalg.vector_norm(filters, ord=2, dim=1)

            total += module.out_channels
            zero += (norms == 0).sum().item()

    return zero / total if total else 0.0

with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

if "actual_filter_sparsity" not in fieldnames:
    fieldnames.append("actual_filter_sparsity")

for row in rows:
    row["actual_filter_sparsity"] = ""

    if row["method"] != "structured":
        continue

    amount = int(float(row["target_sparsity"]) * 100)
    checkpoint = CHECKPOINT_DIR / f"structured_pruning_{amount}.pt"

    model = BaselineCNN()
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)

    value = count_filter_sparsity(model)
    row["actual_filter_sparsity"] = value

    print(
        f"Structured {amount}%: "
        f"{value * 100:.2f}% actual zero filters"
    )

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nUpdated:", csv_path)
print("Training was not repeated.")
