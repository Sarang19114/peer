ye"""
Command-line pipeline to train all peer-matching models and generate
comparison artefacts for stakeholders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.model_comparison import ModelComparator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-model comparison pipeline.")
    parser.add_argument("--output_dir", default="results", help="Directory to store outputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[OK] Training all models...")
    comparator = ModelComparator(output_dir=output_dir)
    results = comparator.train_and_evaluate()

    for idx, (name, res) in enumerate(results.items(), start=1):
        print(f"[{idx}/5] {name}... ok {res.metrics['accuracy']*100:.2f}% accuracy")

    print("\n[OK] Generating reports...")
    comparator.build_metrics_table()
    comparator.build_speed_table()
    comparator.plot_roc_curves()
    comparator.plot_confusion_matrices()
    comparator.plot_performance_radar()
    comparator.generate_pdf_reports()

    print(f"- {output_dir / 'comparison_table.csv'}")
    print(f"- {output_dir / 'roc_curves_comparison.png'}")
    print(f"- {output_dir / 'confusion_matrices_grid.png'}")
    print(f"- {output_dir / 'performance_radar.png'}")
    print(f"- {output_dir / 'Model_Comparison_Summary.pdf'}")


if __name__ == "__main__":
    main()

