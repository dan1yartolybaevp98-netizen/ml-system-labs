import argparse
from config import ensure_dirs
from baseline import run_baseline
from pruning import run_pruning
from distillation import run_distillation
from plot_results import plot_pruning, plot_distillation
from final_report import build_report

def main():
    parser = argparse.ArgumentParser(description="Lab 7: Pruning and Knowledge Distillation")
    parser.add_argument("--stage", choices=["all", "baseline", "pruning", "distillation", "plots", "report"],
                        default="all")
    parser.add_argument("--baseline-epochs", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()

    if args.stage in ("all", "baseline"):
        run_baseline(epochs=args.baseline_epochs) if args.baseline_epochs else run_baseline()
    if args.stage in ("all", "pruning"):
        run_pruning()
    if args.stage in ("all", "distillation"):
        run_distillation()
    if args.stage in ("all", "plots"):
        try:
            plot_pruning()
        except FileNotFoundError as exc:
            print(exc)
        try:
            plot_distillation()
        except FileNotFoundError as exc:
            print(exc)
    if args.stage in ("all", "report"):
        try:
            build_report()
        except FileNotFoundError as exc:
            print(exc)

if __name__ == "__main__":
    main()
