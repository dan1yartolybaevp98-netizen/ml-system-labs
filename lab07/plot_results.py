from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR, ensure_dirs

def plot_pruning():
    path = RESULTS_DIR / "pruning_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run pruning.py first.")
    df = pd.read_csv(path)
    plt.figure(figsize=(9, 6))
    for method in ["global", "layerwise", "structured"]:
        subset = df[df["method"] == method].sort_values("target_sparsity")
        if not subset.empty:
            plt.plot(subset["target_sparsity"] * 100, subset["accuracy_percent"],
                     marker="o", label=method.capitalize())
    plt.xlabel("Target sparsity / filters pruned (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fashion-MNIST: Accuracy vs Pruning Level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = RESULTS_DIR / "accuracy_vs_sparsity.png"
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"Saved {output}")

def plot_distillation():
    path = RESULTS_DIR / "distillation_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run distillation.py first.")
    df = pd.read_csv(path)
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["accuracy_percent"])
    plt.ylabel("Test accuracy (%)")
    plt.title("Plain Student vs Knowledge-Distilled Student")
    plt.xticks(rotation=10)
    plt.tight_layout()
    output = RESULTS_DIR / "student_accuracy_comparison.png"
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"Saved {output}")

if __name__ == "__main__":
    ensure_dirs()
    if (RESULTS_DIR / "pruning_results.csv").exists():
        plot_pruning()
    if (RESULTS_DIR / "distillation_results.csv").exists():
        plot_distillation()
