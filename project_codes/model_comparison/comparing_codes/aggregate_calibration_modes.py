"""
Aggregate calibration_modes.csv files across all backbones x folds and test
whether richer calibration maps (vector / affine) improve held-out ECE over a
single global temperature (scalar).

USAGE
    python aggregate_calibration_modes.py /path/to/results_root

    where results_root contains the per-run output dirs, each holding a
    calibration_modes.csv (as written by evaluate_calibration_modes).

OUTPUT (printed + written to <results_root>/calibration_modes_aggregate.csv):
    - mean ECE per (model, calib_mode)
    - mean delta (scalar - vector), (scalar - affine): positive => richer helps
    - paired Wilcoxon signed-rank over per-run deltas
    - the actual sentence-ready numbers for the manuscript paragraph
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon



def discover(root: Path) -> list[Path]:
    paths = sorted(root.rglob("calibration_modes.csv"))
    if not paths:
        raise FileNotFoundError(f"No calibration_modes.csv under {root}")
    return paths


def load_all(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # run id = parent dir name; lets us pair modes within the same run
        df["run"] = p.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(root_str: str) -> None:
    root = Path(root_str).resolve()
    df = load_all(discover(root))

    # keep only successful fits
    if "fit_ok" in df.columns:
        df = df[df["fit_ok"].fillna(True).astype(bool)]

    # ---- mean ECE / WPC per (model, mode) ----
    summary = (df.groupby(["model", "calib_mode"])
                 .agg(mean_ECE=("ECE", "mean"),
                      sd_ECE=("ECE", "std"),
                      mean_WPC=("WrongPredConfidence", "mean"),
                      n=("ECE", "size"))
                 .reset_index())
    print("\n=== Mean ECE / WrongPredConfidence per (model, mode) ===")
    print(summary.to_string(index=False))

    # ---- paired deltas vs scalar, within each run x model ----
    # pivot so each (run, model) has one column per mode
    piv = df.pivot_table(index=["run", "model"], columns="calib_mode",
                         values="ECE", aggfunc="mean")

    print("\n=== Paired comparison vs scalar temperature (ECE; lower is better) ===")
    print("Positive mean delta (scalar - richer) => richer mode helps.\n")

    results = []
    for richer in ("vector", "affine", "off"):
        if richer not in piv.columns or "scalar" not in piv.columns:
            continue
        sub = piv[["scalar", richer]].dropna()
        if sub.empty:
            continue
        delta = sub["scalar"].to_numpy() - sub[richer].to_numpy()  # >0 => richer lower ECE
        mean_d = float(np.mean(delta))
        n = int(delta.size)
        n_better = int((delta > 0).sum())  # richer beats scalar
        if _HAVE_SCIPY and n >= 1 and np.any(delta != 0):
            try:
                stat, pval = wilcoxon(delta)
                pstr = f"{pval:.3f}"
            except Exception:
                pstr = "n/a"
        else:
            pstr = "n/a (scipy missing or all-zero)"
        results.append(dict(comparison=f"scalar vs {richer}",
                            mean_delta_ECE=mean_d, n_runs=n,
                            n_richer_better=n_better, wilcoxon_p=pstr))
        print(f"  scalar vs {richer:6s}: mean(scalar-{richer}) ECE = {mean_d:+.4f}  "
              f"| richer lower in {n_better}/{n} runs | Wilcoxon p = {pstr}")

    res_df = pd.DataFrame(results)
    out = root / "calibration_modes_aggregate.csv"
    summary.to_csv(out, index=False)
    res_df.to_csv(root / "calibration_modes_paired.csv", index=False)
    print(f"\n[wrote] {out}")
    print(f"[wrote] {root / 'calibration_modes_paired.csv'}")

    # ---- sentence-ready numbers ----
    print("\n=== Manuscript numbers ===")
    sc = summary[summary.calib_mode == "scalar"]["mean_ECE"]
    for richer in ("vector", "affine"):
        rr = summary[summary.calib_mode == richer]["mean_ECE"]
        if not sc.empty and not rr.empty:
            print(f"  mean ECE  scalar={sc.mean():.4f}  {richer}={rr.mean():.4f}")
    # min class support observed (audits the 'few positives' claim)
    if "min_class_support" in df.columns:
        print(f"  min per-class TEST support across all runs: "
              f"{int(df['min_class_support'].min())} "
              f"(median {int(df['min_class_support'].median())})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python aggregate_calibration_modes.py <results_root>")
        raise SystemExit(2)
    main(sys.argv[1])
