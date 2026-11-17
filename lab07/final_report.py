from pathlib import Path
import pandas as pd
from config import RESULTS_DIR

def build_report():
    baseline_file = RESULTS_DIR / "checkpoints" / "baseline_metrics.json"
    pruning_file = RESULTS_DIR / "pruning_results.csv"
    distill_file = RESULTS_DIR / "distillation_results.csv"
    missing = [p.name for p in [baseline_file, pruning_file, distill_file] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment outputs: " + ", ".join(missing))
    import json
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    pruning = pd.read_csv(pruning_file)
    distill = pd.read_csv(distill_file)

    rows = [{
        "Pillar": "Data", "Measurement / evidence": "Fashion-MNIST",
        "Result": "60,000 training images; 10,000 test images; 10 classes"
    }, {
        "Pillar": "Model", "Measurement / evidence": "Baseline CNN",
        "Result": f'{baseline["parameters"]:,} parameters; {baseline["accuracy_percent"]:.2f}% accuracy'
    }]
    best_pruned = pruning.loc[pruning["accuracy_percent"].idxmax()]
    rows.append({
        "Pillar": "Model", "Measurement / evidence": "Best pruning run",
        "Result": f'{best_pruned["method"]}, target {best_pruned["target_sparsity"]:.0%}; '
                  f'{best_pruned["accuracy_percent"]:.2f}% accuracy; '
                  f'{best_pruned["actual_sparsity"]:.1%} actual zero weights'
    })
    for _, r in distill.iterrows():
        rows.append({
            "Pillar": "Model", "Measurement / evidence": r["model"],
            "Result": f'{r["parameters"]:,} parameters; {r["accuracy_percent"]:.2f}% accuracy; '
                      f'{r["size_mb"]:.3f} MB'
        })
    rows.append({
        "Pillar": "Compute", "Measurement / evidence": "Baseline inference",
        "Result": f'{baseline["latency_ms_per_image_batch_1"]:.3f} ms/image (batch size 1), '
                  f'device {baseline["device"]}'
    })
    rows.append({
        "Pillar": "Compute", "Measurement / evidence": "Pruning and student latency",
        "Result": "See pruning_results.csv and distillation_results.csv; measured on the recorded device"
    })
    rows.append({
        "Pillar": "Energy", "Measurement / evidence": "Direct energy measurement",
        "Result": "Not measured. No energy values are fabricated; energy efficiency is discussed qualitatively."
    })
    out = RESULTS_DIR / "final_comparison_table.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved final table: {out}")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    build_report()
