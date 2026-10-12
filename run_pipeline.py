"""
run_pipeline.py
===============

Runs the replication pipeline stage by stage. Every script in project_codes/
resolves its inputs and outputs through paths relative to its own directory
(e.g. ``../../project_data/``), so each step is executed with its own directory
as the working directory.

    python run_pipeline.py                 # all stages, in order
    python run_pipeline.py data tda        # only the named stages, in the order given
    python run_pipeline.py --list          # list stages and their steps

Stages (in order):
    data      build the 182 x 66 panel from PaxPetrolica_Main.dta
    tda       persistence diagrams, functional summaries, PD-kernel similarity figures
    bare      train + 5-fold CV of the ten bare (no-TDA) backbones, then CV figures
    tda_model train + 5-fold CV of the ten TDA-conditioned backbones, then CV figures
              and PDDeepSets / topological-prior diagnostics
    compare   paired bare-vs-TDA comparison: metrics, calibration, feature-importance
              ranks, LaTeX tables, pooled Wilcoxon tests, conditioning-attention figures

The ``bare`` and ``tda_model`` stages train 10 backbones x 5 folds for up to
900 epochs each.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODES = ROOT / "project_codes"

# stage -> list of (working directory, script, arguments)
STAGES = {
    "data": [
        ("data_preparation", "data_processing_class_pastonly.py", []),
        ("data_preparation", "distribution_check_pastonly.py", []),
    ],
    "tda": [
        ("persistence_diagrams", "run_all_gudhi_scripts.py", []),
        ("persistence_diagrams", "tda_precompute_kernels.py", []),
    ],
    "bare": [
        ("bare_classifier", "main_classification.py", []),
        ("bare_classifier", "finalize_all.py", []),
        ("bare_classifier", "fold_class_support.py", []),
    ],
    "tda_model": [
        ("tda_classifier", "main.py", ["classify", "finalize_all", "pd_deepsets", "fold_support"]),
    ],
    "compare": [
        ("model_comparison", "compare_all.py", ["all"]),
        ("model_comparison", "compare_all.py", ["tables"]),
        ("model_comparison", "compare_all.py", ["stats"]),
        ("model_comparison", "aggregate_beta_attention.py", []),
    ],
}


def run_step(subdir: str, script: str, args: list[str]) -> None:
    cwd = CODES / subdir
    cmd = [sys.executable, script, *args]
    print(f"\n[run_pipeline] ({subdir}) {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str]) -> None:
    if any(a in ("--list", "-l", "--help", "-h") for a in argv):
        for name, steps in STAGES.items():
            print(name)
            for subdir, script, args in steps:
                print(f"    {subdir}/{script} {' '.join(args)}".rstrip())
        return

    selected = argv or list(STAGES)
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(unknown)}. Valid: {', '.join(STAGES)}")

    raw = ROOT / "project_data" / "PaxPetrolica_Main.dta"
    if "data" in selected and not raw.exists():
        sys.exit(f"Missing {raw}. See project_data/README.md for the download.")

    for name in selected:
        print("\n" + "=" * 70 + f"\n[run_pipeline] stage: {name}\n" + "=" * 70)
        for step in STAGES[name]:
            run_step(*step)
    print("\n[run_pipeline] done.")


if __name__ == "__main__":
    main(sys.argv[1:])
