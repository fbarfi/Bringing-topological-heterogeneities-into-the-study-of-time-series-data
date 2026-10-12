"""
compare_all.py
==========

Consolidated, single-entry-point bare-vs-TDA comparison pipeline for the
revisionist-dispute-onset classification study.  One well-structured tool 
the subcommands of which share a single source of truth for configuration, 
fold construction, and result-tree paths.

SUBCOMMANDS
  engine     Per-fold full-engine comparison (calibration, ECE, McNemar,
             bootstrap CIs, change-point detection, abstention) via the intact
             compare_with_args, run per fold, then fold-mean aggregated into the
             model-level files ReportRunner expects, plus feature-importance
             fold-means/plots, and all plots. Also pools the per-fold
             calibration-mode results into the `calibration` outputs (below).

  importance Build feature-importance fold-means and plots only, from the
             existing per-fold feature_importance_*.csv files in BOTH results
             trees (TDA and no-TDA), without re-running the engine. Writes
             per-tree mean(+sd) CSVs/plots and TDA-vs-no-TDA overlay plots
             (gradient, perturbation, and permutation) into the OUT_ROOT
             model dirs. Also writes across-backbone, scale-free rank-comparison
             plots (importance_rank_<metric>_<split>_tda_vs_notda.png) into
             OUT_ROOT: within each backbone the covariates are ranked by
             importance, and the per-covariate rank distribution across all
             backbones is compared TDA vs no-TDA -- a population-level view that
             replaces the single-backbone illustration.

  plots      Regenerate the summary/reliability/CPD figures from existing
             per-fold and per-sample CSVs (no engine re-run), via
             ReportRunner in comparing_codes.plot_summary. Includes the
             per-family reliability overlays (all TDA backbones, all base
             backbones) and the TDA-vs-base across-backbone reliability spread
             plot, written to the summary_plots/calibration directory.

  pooled     Engine-free paired comparison built directly from each the
             cached fold_predictions.npz of each fold: pooled out-of-fold predictions 
             (each country held out exactly once), exact McNemar, quadratic-weighted
             kappa, and macro-AUC delta with a paired bootstrap 95% CI. Emits a
             publication table (.csv and .tex).

  tables     Build all three LaTeX performance tables -- TDA performance,
             no-TDA performance, and the paired Base-vs-TDA overfitting table --
             from the SAME pair of per-model summary CSVs, so shared cells can
             never disagree across tables.

  stats      Headline pooled Wilcoxon signed-rank + across-backbone sign tests:
             pool fold-level TDA-Base deltas across all backbones
             (10 backbones x 5 folds = 50 obs/metric), one test per metric.
             Reads OUT_ROOT/<backbone>/per_fold_summary.csv (needs columns
             metric, delta, higher_is_better, written by the engine route).
             Emits pooled_wilcoxon_results.{csv,tex}. The overfitting analysis
             lives in the `tables` subcommand.

  calibration  Pool the per-fold calibration_modes.csv files (written by the
             engine route) across all backbones x folds and test whether a
             richer post-hoc calibration map improves held-out calibration over
             a single global temperature. For each fold, a calibrator is fit on
             the validation split and applied to the held-out test fold under
             four modes -- none (raw), scalar temperature, per-class temperature,
             and affine -- and macro-OvR ECE is recomputed for each. Reports the
             mean ECE per mode (base and TDA) and the paired scalar-vs-richer
             ECE deltas with Wilcoxon signed-rank tests, alongside the per-class
             held-out support that bounds how reliably per-class parameters can
             be fit. Emits calibration_modes_summary.csv,
             calibration_modes_paired.csv, and calibration_modes.tex. Runs
             automatically at the end of `engine` and `all`; can also be invoked
             standalone (no engine re-run) once the per-fold files exist.

  all        Run `engine` (incl. calibration pooling), then `pooled`.


PAIRING SAFETY
  Folds are paired by construction (same StratifiedKFold seed, same y order, in
  both result trees). The pooled route additionally VERIFIES pairing at run time
  by asserting the two pooled label vectors are identical.

USAGE
  python compare_all.py engine
  python compare_all.py importance
  python compare_all.py plots
  python compare_all.py pooled
  python compare_all.py tables
  python compare_all.py stats
  python compare_all.py calibration
  python compare_all.py all
  # optional overrides (see each subcommand's --help), e.g.:
  python compare_all.py pooled --base-root DIR --tda-root DIR --out-dir DIR --n-classes 4
  python compare_all.py tables --tda-summary CSV --base-summary CSV --out-dir DIR
  python compare_all.py stats  --cv-root DIR --out-dir DIR
  python compare_all.py calibration --cv-root DIR --out-dir DIR
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse, traceback 
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from scipy.stats import binomtest
from scipy.stats import t as _tdist
from scipy import stats as _st
from scipy import stats

from comparing_codes.comparing_tsai_models import compare_with_args
from comparing_codes.plot_summary import ReportRunner, PlotConfig


# ============================== CONFIG ==============================
# Single source of truth for every constant the routes share.

HERE = Path(__file__).resolve().parent

NEPOCHS   = 900
N_SPLITS  = 5
SEED      = 42
VAL_FRAC  = 0.15        # validation carved out of each fold's training portion
N_CLASSES = 4

SPLIT     = "test"
VAL_SIZE  = 0.2         # naming tag only (matches the trained-model dir layout)
TEST_SIZE = 0.1         # naming tag only (matches the trained-model dir layout)

TSAI_MODELS = [
    "RNNAttention", "LSTMAttention", "GRUAttention", "RNNPlus",
    "TSSequencerPlus", "TSiT", "TSiTPlus", "ResNetPlus",
    "InceptionTimePlus", "XCMPlus",
]

features = ['revleader', 'new_oil_state', 'coldwar', 'majorpower', 'logoil',
            'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity']
tda_features = ["Amplitude", "Entropy", "L1", "C1", "L2", "pc1"]

# Result / code trees (resolved relative to this script's location).
BASE_RESULTS_ROOT = (HERE / "../../project_results/tsai_no_tda_Results/classification_models").resolve()
TDA_RESULTS_ROOT  = (HERE / "../../project_results/tsai_yes_tda_gpd_Results/classification_models").resolve()
BASE_CODE_ROOT    = (HERE / "../../project_codes/bare_classifier").resolve()
TDA_CODE_ROOT     = (HERE / "../../project_codes/tda_classifier").resolve()
OUT_ROOT          = (HERE / f"../../project_results/tsai_comparing_models_gpd_results/nepochs_{NEPOCHS}/{SPLIT}_cv").resolve()
POOLED_OUT_ROOT   = (HERE / f"../../project_results/tsai_comparing_models_gpd_results/nepochs_{NEPOCHS}/cv_pooled").resolve()
DV_PATH           = (HERE / "../../project_data/dv_augm.h5").resolve()

# Per-model performance summaries (one row per backbone, train/valid/test
# accuracy and AUROC means + stds) that the LaTeX tables are built from. These
# are produced by the training/eval pipeline, NOT by compare_all.py; the `tables`
# subcommand reads this exact pair so all emitted tables are mutually consistent.
TDA_SUMMARY_CSV   = (TDA_RESULTS_ROOT / "all_models_performance_summary_900.csv").resolve()
BASE_SUMMARY_CSV  = (BASE_RESULTS_ROOT / "all_models_performance_summary_900.csv").resolve()
TABLES_OUT_DIR    = (OUT_ROOT / "tables").resolve()

PD_FOLDER = "../../project_results/gudhi_multiple_tda_results_H_4/PersistenceDiagrams/"
HIDX      = str(1234)

CALIB_OUT_DIR     = (OUT_ROOT / "calibration").resolve()


# ============================ SHARED HELPERS ============================

def load_y() -> np.ndarray:
    """Per-country label = max revonsets category (collapsing >2 into class 3)."""
    dv = pd.read_hdf(DV_PATH, key="data").sort_values(["ccode", "year"])
    dv["revonsets_category"] = dv["revonsets"]
    dv.loc[dv["revonsets"] > 2, "revonsets_category"] = 3
    dv["revonsets"] = dv["revonsets_category"].astype(int)
    dv = dv.drop(columns=["revonsets_category"])
    ccodes = dv["ccode"].unique().tolist()
    return np.array([dv.loc[dv["ccode"] == c, "revonsets"].max() for c in ccodes])


def make_folds(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rebuild the SAME (train, valid, test) country folds used in training."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = []
    for tr_full, te in skf.split(np.zeros(len(y)), y):
        tr, va = train_test_split(tr_full, test_size=VAL_FRAC,
                                  stratify=y[tr_full], random_state=SEED)
        folds.append((np.asarray(tr), np.asarray(va), np.asarray(te)))
    return folds


def pkl_path(root: Path, model: str, fold: int) -> Path:
    return (root / f"epochs_{NEPOCHS}" / f"{model}_val_{VAL_SIZE}_test_{TEST_SIZE}"
            / f"fold_{fold}" / "models" / "trained_classification_model.pkl").resolve()

def _calibration_modes_table(summary: "pd.DataFrame",
                             paired: "pd.DataFrame") -> str:
    """LaTeX table for the calibration-mode comparison. Counts are read from the
    data, not hard-coded, so the caption can't drift from what was aggregated."""
    mode_order = ["off", "scalar", "vector", "affine"]
    pretty = {"off": "None (raw)", "scalar": "Scalar T",
              "vector": "Per-class T", "affine": "Affine"}

    # actual per-cell n (folds x backbones) and paired n, from the data
    per_cell_n = int(summary["n"].max()) if "n" in summary.columns and len(summary) else 0
    paired_n = int(paired["n_runs"].max()) if "n_runs" in paired.columns and len(paired) else 0

    def _ece(model, mode):
        r = summary[(summary["model"] == model) & (summary["calib_mode"] == mode)]
        return f"{r['mean_ECE'].iloc[0]:.4f}" if len(r) else "--"

    top_lines = []
    for mode in mode_order:
        top_lines.append(
            f"    {pretty.get(mode, mode)} & {_ece('base', mode)} & "
            f"{_ece('tda', mode)} \\\\")
    top_body = "\n     \\hline\n".join(top_lines)

    bot_lines = []
    has_model = "model" in paired.columns
    for _, row in paired.iterrows():
        comp = pretty.get(str(row["comparison"]).replace("scalar vs ", ""),
                          str(row["comparison"]))
        model_lbl = str(row["model"]).upper() if has_model else ""
        label = f"Scalar vs {comp} ({model_lbl})" if has_model else f"Scalar vs {comp}"
        md = f"{row['mean_delta_ECE']:+.4f}"
        nb = f"{int(row['n_richer_better'])}/{int(row['n_runs'])}"
        p = row["wilcoxon_p"]
        p = f"{float(p):.3f}" if str(p).replace('.', '', 1).isdigit() else str(p)
        bot_lines.append(f"    {label} & {md} & {nb} & {p} \\\\")
    bot_body = "\n     \\hline\n".join(bot_lines)

    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Held-out calibration (macro-OvR ECE) under increasing "
        "post-hoc calibration flexibility, fit per fold on the validation split "
        "and applied to the held-out test fold. Top: mean ECE per mode "
        f"(each averaged over {per_cell_n} held-out folds). Bottom: paired "
        "comparison of each richer map against a single global temperature, "
        f"reported separately for each model over {paired_n} paired runs "
        "(backbones $\\times$ folds) per model; a \\emph{{negative}} $\\Delta$ "
        "(scalar~$-$~richer) means the richer map has HIGHER ECE, i.e.\\ is worse, "
        "and the count reports how many runs the richer map improves on scalar. "
        "Smaller ECE is better.}\n"
        "  \\label{tab:calibration-modes}\n"
        "  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{6pt}\n"
        "  \\begin{tabular}{l | r r}\n"
        "    \\hline\\hline\n"
        "    \\textbf{Calibration} & \\textbf{Base ECE} & \\textbf{TDA ECE} \\\\\n"
        "    \\hline\\hline\n"
        f"{top_body}\n"
        "     \\hline\\hline\n"
        "  \\end{tabular}\n"
        "  \\vspace{4pt}\n"
        "  \\begin{tabular}{l | r r r}\n"
        "    \\hline\\hline\n"
        "    \\textbf{Comparison} & \\textbf{Mean $\\Delta$ECE} & "
        "\\textbf{Richer better} & \\textbf{Wilcoxon $p$} \\\\\n"
        "    \\hline\\hline\n"
        f"{bot_body}\n"
        "     \\hline\\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def run_calibration(cv_root: str, out_dir: str) -> None:
    """Pool per-fold calibration_modes.csv across all backbones and test whether
    richer calibration maps (per-class temperature 'vector', or 'affine') beat a
    single global temperature on held-out ECE. Mirrors run_stats' structure."""
    cv_root = Path(cv_root)
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(str(cv_root), "**", "calibration_modes.csv"),
                             recursive=True))
    if not paths:
        print(f"[calib] no calibration_modes.csv under {cv_root} — run `engine` "
              f"first (with the evaluate_calibration_modes hook in "
              f"CompareRunner.run).")
        return
    print(f"[calib] found {len(paths)} per-fold calibration_modes.csv files")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # run id (backbone+fold) = the fold dir's parent chain; use relative path
        df["run"] = os.path.relpath(os.path.dirname(p), str(cv_root))
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)

    if "fit_ok" in allf.columns:
        allf = allf[allf["fit_ok"].fillna(True).astype(bool)]

    # ---- per (model, mode) means
    summary = (allf.groupby(["model", "calib_mode"])
                   .agg(mean_ECE=("ECE", "mean"),
                        sd_ECE=("ECE", "std"),
                        mean_WPC=("WrongPredConfidence", "mean"),
                        n=("ECE", "size"))
                   .reset_index())
    sum_path = os.path.join(out_dir, "calibration_modes_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"[calib] wrote {sum_path}")

    # ---- paired scalar-vs-richer within each (run, model)
    # Test each model separately: with B backbones x K folds runs, each model
    # contributes exactly one paired delta per run (n = B*K, e.g. 10x5 = 50).
    # Stacking base and TDA into a single test would double n (e.g. to 100) and
    # treat the same folds twice under two models, which is not a valid paired
    # sample; we therefore keep the two models as separate rows.
    piv = allf.pivot_table(index=["run", "model"], columns="calib_mode",
                           values="ECE", aggfunc="mean")
    rows = []
    for richer in ("vector", "affine", "off"):
        if richer not in piv.columns or "scalar" not in piv.columns:
            continue
        for model in ("base", "tda"):
            try:
                sub = piv.xs(model, level="model")[["scalar", richer]].dropna()
            except KeyError:
                continue
            if sub.empty:
                continue
            delta = sub["scalar"].to_numpy() - sub[richer].to_numpy()  # >0 => richer lower ECE
            n = int(delta.size)
            n_better = int((delta > 0).sum())
            try:
                _, pval = stats.wilcoxon(delta)
                pstr = f"{pval:.3f}"
            except Exception:
                pstr = "n/a"
            rows.append(dict(comparison=f"scalar vs {richer}", model=model,
                             mean_delta_ECE=float(np.mean(delta)),
                             n_runs=n, n_richer_better=n_better,
                             wilcoxon_p=pstr))
    paired = pd.DataFrame(rows)
    paired_path = os.path.join(out_dir, "calibration_modes_paired.csv")
    paired.to_csv(paired_path, index=False)
    print(f"[calib] wrote {paired_path}")

    # ---- LaTeX
    tex_path = os.path.join(out_dir, "calibration_modes.tex")
    with open(tex_path, "w") as fh:
        fh.write(_calibration_modes_table(summary, paired))
    print(f"[calib] wrote {tex_path}")

    # ---- console: sentence-ready numbers
    def _m(model, mode):
        r = summary[(summary.model == model) & (summary.calib_mode == mode)]
        return float(r["mean_ECE"].iloc[0]) if len(r) else float("nan")
    print("\n[calib] mean ECE (base / tda):")
    for mode in ("off", "scalar", "vector", "affine"):
        print(f"    {mode:7s}: {_m('base', mode):.4f} / {_m('tda', mode):.4f}")
    if "min_class_support" in allf.columns:
        print(f"[calib] min per-class TEST support across runs: "
              f"{int(allf['min_class_support'].min())} "
              f"(median {int(allf['min_class_support'].median())})")
    print("[calib] calibration-mode comparison complete.")

# ============================ ENGINE ROUTE ============================

def _model_dir(model: str) -> Path:
    return (OUT_ROOT / f"{model}_val_{VAL_SIZE}_test_{TEST_SIZE}_epochs_{NEPOCHS}").resolve()


def _fold_out(model: str, fold: int) -> Path:
    return (_model_dir(model) / f"fold_{fold}").resolve()


def _args_for_fold(fold_split) -> Dict[str, Any]:
    return {
        "BASE_CODE": str(BASE_CODE_ROOT), "TDA_CODE": str(TDA_CODE_ROOT),
        "SPLIT": SPLIT, "features": features, "tda_features": tda_features,
        "POS_LABEL": 1, "THRESHOLD": None,
        "N_BOOT": 2000, "ALPHA": 0.05, "RANDOM_SEED": SEED,
        "val_size": VAL_SIZE, "test_size": TEST_SIZE,
        "EVENTS": [10, 22, 45], "CPD_TOLERANCE": 2, "CPD_MIN_SEP": 2,
        "CALIB": "scalar", "ABSTAIN_THRESH": None, "ECE_BINS": 15,
        "n_years": list(range(1945, 2011)),
        "FOLD_SPLIT": (fold_split[0].tolist(),
                       fold_split[1].tolist(),
                       fold_split[2].tolist()),
        "pd_folder": PD_FOLDER,
        "Hidx": HIDX,
    }


def _run_one_fold(model: str, fold: int, fold_split) -> Optional[Dict[str, Any]]:
    base_pkl = pkl_path(BASE_RESULTS_ROOT, model, fold)
    tda_pkl  = pkl_path(TDA_RESULTS_ROOT,  model, fold)
    if not base_pkl.exists() or not tda_pkl.exists():
        print(f"[skip] {model} fold {fold}: missing pkl "
              f"(base={base_pkl.exists()}, tda={tda_pkl.exists()})")
        return None
    out_dir = _fold_out(model, fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = compare_with_args(_args_for_fold(fold_split),
                                 str(base_pkl), str(tda_pkl), str(out_dir))
    try:
        with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"[warn] manifest write failed: {e}")
    return manifest


def _mean_csv(paths, out_path) -> bool:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        return False
    base = frames[0].copy()
    num_cols = base.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        stacked = np.stack([f[num_cols].to_numpy(dtype=float) for f in frames], 0)
        base[num_cols] = stacked.mean(0)
    base.to_csv(out_path, index=False)
    return True


def _concat_csv(paths, out_path) -> bool:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        return False
    pd.concat(frames, axis=0, ignore_index=True).to_csv(out_path, index=False)
    return True


def _mean_cpd(paths, out_path) -> bool:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        return False
    f0 = frames[0]
    key = "year" if "year" in f0.columns else f0.columns[0]
    val_cols = [c for c in f0.columns if c != key]
    merged = None
    for i, f in enumerate(frames):
        f = f[[key] + val_cols].rename(columns={c: f"{c}__{i}" for c in val_cols})
        merged = f if merged is None else merged.merge(f, on=key, how="outer")
    out = pd.DataFrame({key: merged[key]})
    for c in val_cols:
        cols = [f"{c}__{i}" for i in range(len(frames)) if f"{c}__{i}" in merged]
        out[c] = merged[cols].mean(axis=1)
    out.sort_values(key).to_csv(out_path, index=False)
    return True


def _mean_importance_long(paths, out_path,
                          group_keys=("Feature", "Dataset"),
                          value_col="Importance") -> bool:
    """Fold-mean + fold-std for a LONG-format importance file, e.g.
    feature_importance_scores.csv with columns (Feature, <value>, Dataset).
    Aligns on group_keys (so Train/Test rows stay separate and feature order is
    irrelevant), then averages the value column across folds. The value column
    defaults to `value_col` but, if that name is absent, falls back to the sole
    remaining numeric column (handles trees that name it differently)."""
    frames = []
    for i, p in enumerate(paths):
        if os.path.exists(p):
            f = pd.read_csv(p)
            f["__fold"] = i
            frames.append(f)
    if not frames:
        return False
    allf = pd.concat(frames, axis=0, ignore_index=True)
    keys = [k for k in group_keys if k in allf.columns]

    # Resolve the value column robustly.
    if value_col not in allf.columns:
        numeric = [c for c in allf.columns
                   if c not in keys and c != "__fold"
                   and pd.api.types.is_numeric_dtype(allf[c])]
        if not numeric:
            print(f"[imp] [warn] {os.path.basename(out_path)}: no numeric value "
                  f"column found (cols={list(allf.columns)}); skipping")
            return False
        value_col = numeric[0]

    g = allf.groupby(keys, dropna=False)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value_col}_mean",
                              "std":  f"{value_col}_std",
                              "count": "n_folds"})
    out.to_csv(out_path, index=False)
    return True


def _mean_importance_wide(paths, out_path, key="Feature") -> bool:
    """Fold-mean + fold-std for a WIDE-format importance file, e.g.
    feature_importance_gradients_and_perturbations.csv with one row per feature
    and several numeric importance columns. Aligns on the feature key and
    averages each numeric column across folds; emits <col>_mean and <col>_std
    per column."""
    frames = []
    for i, p in enumerate(paths):
        if os.path.exists(p):
            f = pd.read_csv(p)
            f["__fold"] = i
            frames.append(f)
    if not frames:
        return False
    allf = pd.concat(frames, axis=0, ignore_index=True)
    if key not in allf.columns:
        key = allf.columns[0]
    val_cols = [c for c in allf.columns
                if c not in (key, "__fold")
                and pd.api.types.is_numeric_dtype(allf[c])]
    dropped = [c for c in allf.columns
               if c not in (key, "__fold") and c not in val_cols]
    if dropped:
        print(f"[imp] [warn] {os.path.basename(out_path)}: non-numeric columns "
              f"skipped (not averaged): {dropped}")
    if not val_cols:
        print(f"[imp] [warn] {os.path.basename(out_path)}: no numeric importance "
              f"columns found (cols={list(allf.columns)})")
        return False
    g = allf.groupby(key, dropna=False)
    out = pd.DataFrame({key: sorted(allf[key].unique())}).set_index(key)
    means = g[val_cols].mean()
    stds = g[val_cols].std()
    for c in val_cols:
        out[f"{c}_mean"] = means[c]
        out[f"{c}_std"] = stds[c]
    out["n_folds"] = g.size()
    out.reset_index().to_csv(out_path, index=False)
    return True


def _mean_importance_hybrid(paths, out_path, key="Feature", dataset="Dataset") -> bool:
    """Fold-mean + fold-std for a HYBRID-format importance file that has BOTH a
    Dataset column (Train/Test/...) AND several numeric value columns, e.g. the
    no-TDA feature_importance_gradients_and_perturbations.csv with columns
    (Feature, Gradient Importance, Perturbation Importance, Dataset).

    Groups by (Feature, Dataset) and averages EVERY numeric value column across
    folds, emitting <col>_mean / <col>_std per value column; no value column is
    dropped."""
    frames = []
    for i, p in enumerate(paths):
        if os.path.exists(p):
            f = pd.read_csv(p)
            f["__fold"] = i
            frames.append(f)
    if not frames:
        return False
    allf = pd.concat(frames, axis=0, ignore_index=True)
    keys = [k for k in (key, dataset) if k in allf.columns]
    val_cols = [c for c in allf.columns
                if c not in keys and c != "__fold"
                and pd.api.types.is_numeric_dtype(allf[c])]
    if not val_cols:
        print(f"[imp] [warn] {os.path.basename(out_path)}: no numeric value "
              f"columns (cols={list(allf.columns)})")
        return False
    g = allf.groupby(keys, dropna=False)
    means = g[val_cols].mean()
    stds = g[val_cols].std()
    out = means.copy()
    out.columns = [f"{c}_mean" for c in val_cols]
    for c in val_cols:
        out[f"{c}_std"] = stds[c]
    out["n_folds"] = g.size()
    out.reset_index().to_csv(out_path, index=False)
    return True


def _plot_importance_hybrid(cv_csv, out_png, model) -> bool:
    """Plot a HYBRID *_cv.csv (Feature, Dataset, <col>_mean/<col>_std ...): one
    row of panels per value-metric, one column per Dataset, bars sorted, with
    std error bars."""
    if not os.path.exists(cv_csv):
        return False
    df = pd.read_csv(cv_csv)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols or "Dataset" not in df.columns:
        return False
    datasets = list(df["Dataset"].unique())
    nrow, ncol = len(mean_cols), len(datasets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    for r, mcol in enumerate(mean_cols):
        base = mcol[:-5]
        scol = base + "_std"
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            sub = df[df["Dataset"] == ds].sort_values(mcol)
            err = sub[scol].fillna(0.0) if scol in sub.columns else None
            ax.barh(sub["Feature"], sub[mcol], xerr=err,
                    color="#4C72B0", ecolor="#333333", capsize=3, alpha=0.9)
            ax.axvline(0.0, color="#888888", lw=0.8)
            ax.set_title(f"{model} — {base} ({ds})", fontsize=9)
            ax.set_xlabel(f"{base} (fold mean $\\pm$ sd)", fontsize=8)
            ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _aggregate_model_folds(model: str) -> bool:
    """Build top-level summary.csv / per_sample_predictions.csv / per_class /
    cpd_*.csv / feature-importance means from the model's fold_*/ outputs,
    where ReportRunner expects them."""
    mdir = _model_dir(model)
    folds = sorted(glob.glob(os.path.join(mdir, "fold_*")),
                   key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not folds:
        return False
    ok = _mean_csv([os.path.join(f, "summary.csv") for f in folds],
                   os.path.join(mdir, "summary.csv"))
    _mean_csv([os.path.join(f, "per_class_auroc_auprc.csv") for f in folds],
              os.path.join(mdir, "per_class_auroc_auprc.csv"))
    _concat_csv([os.path.join(f, "per_sample_predictions.csv") for f in folds],
                os.path.join(mdir, "per_sample_predictions.csv"))
    cpd_names = set()
    for f in folds:
        for p in glob.glob(os.path.join(f, "cpd_*.csv")):
            cpd_names.add(os.path.basename(p))
    for name in sorted(cpd_names):
        _mean_cpd([os.path.join(f, name) for f in folds],
                  os.path.join(mdir, name))
    # feature-importance fold-means (+ std for error bars)
    _mean_importance_long(
        [os.path.join(f, "feature_importance_scores.csv") for f in folds],
        os.path.join(mdir, "feature_importance_scores_cv.csv"),
        group_keys=("Feature", "Dataset"), value_col="Importance")
    _mean_importance_wide(
        [os.path.join(f, "feature_importance_gradients_and_perturbations.csv") for f in folds],
        os.path.join(mdir, "feature_importance_gradients_and_perturbations_cv.csv"),
        key="Feature")
    fs = [pd.read_csv(os.path.join(f, "summary.csv")).assign(fold=i)
          for i, f in enumerate(folds)
          if os.path.exists(os.path.join(f, "summary.csv"))]
    if fs:
        allf = pd.concat(fs, axis=0, ignore_index=True)
        allf.to_csv(mdir / "per_fold_summary.csv", index=False)
        num = allf.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
        num.agg(["mean", "std"]).T.rename(columns={"mean": "mean", "std": "std"}) \
           .to_csv(mdir / "cv_mean_sd_summary.csv")
    return ok

def _plot_importance_long(cv_csv: str, out_png: str, model: str) -> bool:
    """Horizontal bar chart of fold-mean perturbation importance with cross-fold
    std error bars, one panel per Dataset (Train/Test/...), from the long-format
    *_cv.csv produced by _mean_importance_long. The value column name is
    auto-detected (the sole '*_mean' column), so it need not be 'Importance'."""
    if not os.path.exists(cv_csv):
        return False
    df = pd.read_csv(cv_csv)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols:
        return False
    mcol = mean_cols[0]
    scol = mcol[:-5] + "_std"
    base_label = mcol[:-5]
    has_std = scol in df.columns
    datasets = list(df["Dataset"].unique()) if "Dataset" in df.columns else [None]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["Dataset"] == ds] if ds is not None else df
        sub = sub.sort_values(mcol)
        err = sub[scol].fillna(0.0) if has_std else None
        ax.barh(sub["Feature"], sub[mcol], xerr=err,
                color="#4C72B0", ecolor="#333333", capsize=3, alpha=0.9)
        ax.axvline(0.0, color="#888888", lw=0.8)
        ax.set_title(f"{model} — {ds}" if ds is not None else model, fontsize=10)
        ax.set_xlabel(f"{base_label} (fold mean $\\pm$ sd)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_importance_wide(cv_csv: str, out_png: str, model: str) -> bool:
    """Horizontal bar charts of fold-mean importance with std error bars, one
    panel per <col>_mean found in the wide-format *_cv.csv produced by
    _mean_importance_wide."""
    if not os.path.exists(cv_csv):
        return False
    df = pd.read_csv(cv_csv)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols:
        return False
    n = len(mean_cols)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 3.6 * nrow),
                             squeeze=False)
    flat = axes.flatten()
    for ax, mc in zip(flat, mean_cols):
        base = mc[:-5]  # strip "_mean"
        sc = f"{base}_std"
        sub = df[["Feature", mc] + ([sc] if sc in df.columns else [])].copy()
        sub = sub.sort_values(mc)
        err = sub[sc].fillna(0.0) if sc in sub.columns else None
        ax.barh(sub["Feature"], sub[mc], xerr=err,
                color="#55A868", ecolor="#333333", capsize=3, alpha=0.9)
        ax.axvline(0.0, color="#888888", lw=0.8)
        ax.set_title(f"{model} — {base}", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _importance_fold_dirs(results_root: Path, model: str) -> list:
    """Per-fold dirs that hold the training-time feature_importance_*.csv files,
    in a results tree. Mirrors pkl_path's layout:
      <results_root>/epochs_<N>/<model>_val_<v>_test_<t>/fold_<k>/
    (NOTE: epochs_<N> is a PARENT level and the model dir has no _epochs suffix,
    unlike the comparison OUT_ROOT model dirs.)"""
    base = (results_root / f"epochs_{NEPOCHS}"
            / f"{model}_val_{VAL_SIZE}_test_{TEST_SIZE}").resolve()
    return sorted(glob.glob(os.path.join(base, "fold_*")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))


# results trees to draw per-fold importance from -> output tag for filenames
IMPORTANCE_SOURCES = [
    ("tda",   TDA_RESULTS_ROOT),
    ("notda", BASE_RESULTS_ROOT),
]


def _discover_importance_basenames(folds) -> set:
    """Find every per-fold importance CSV basename present under the folds,
    matching feature_importance_*.csv (so we are not tied to two fixed names)."""
    names = set()
    for f in folds:
        for p in glob.glob(os.path.join(f, "feature_importance_*.csv")):
            b = os.path.basename(p)
            if b.endswith("_cv.csv"):
                continue
            names.add(b)
    return names


def plot_importance_means(verbose: bool = True) -> None:
    """For every backbone and BOTH results trees (TDA and no-TDA), render
    fold-mean (+sd) feature-importance plots.

    Per-fold importance files are read from the RESULTS trees (where training
    wrote them, next to the .pkl / fold_predictions.npz). The aggregated
    *_cv.csv and *_cv.png are written into the comparison OUT_ROOT model dir,
    with a tree tag (tda / notda) in the filename so the two trees produce
    SEPARATE files. Importance filenames are auto-discovered by glob; long- vs
    wide-format is auto-detected (a 'Dataset' column => long)."""
    any_plot = False
    for model in TSAI_MODELS:
        out_mdir = _model_dir(model)
        out_mdir.mkdir(parents=True, exist_ok=True)

        for tag, results_root in IMPORTANCE_SOURCES:
            folds = _importance_fold_dirs(results_root, model)
            if not folds:
                if verbose:
                    print(f"[imp] {model} [{tag}]: no fold dirs under "
                          f"{(results_root / f'epochs_{NEPOCHS}').resolve()} — skip")
                continue
            basenames = _discover_importance_basenames(folds)
            if not basenames:
                if verbose:
                    print(f"[imp] {model} [{tag}]: no feature_importance_*.csv "
                          f"in {len(folds)} fold dirs — skip")
                continue

            for base in sorted(basenames):
                stem = base[:-4]                       # strip ".csv"
                cv_csv = os.path.join(out_mdir, f"{stem}_{tag}_cv.csv")
                png = os.path.join(out_mdir, f"{stem}_{tag}_cv.png")
                fold_files = [os.path.join(f, base) for f in folds]
                present = [p for p in fold_files if os.path.exists(p)]
                if not present:
                    continue

                sample = pd.read_csv(present[0])
                numeric = [c for c in sample.columns
                           if c not in ("Feature", "Dataset")
                           and pd.api.types.is_numeric_dtype(sample[c])]
                if "Dataset" in sample.columns and len(numeric) >= 2:
                    # hybrid: Dataset column AND several value columns
                    _mean_importance_hybrid(fold_files, cv_csv,
                                            key="Feature", dataset="Dataset")
                    ok = _plot_importance_hybrid(cv_csv, png, f"{model} [{tag}]")
                elif "Dataset" in sample.columns:
                    _mean_importance_long(fold_files, cv_csv,
                                          group_keys=("Feature", "Dataset"),
                                          value_col="Importance")
                    ok = _plot_importance_long(cv_csv, png, f"{model} [{tag}]")
                else:
                    _mean_importance_wide(fold_files, cv_csv, key="Feature")
                    ok = _plot_importance_wide(cv_csv, png, f"{model} [{tag}]")

                if ok:
                    any_plot = True
                    if verbose:
                        print(f"[imp] {model} [{tag}]: wrote {os.path.basename(png)} "
                              f"(from {len(present)}/{len(folds)} folds)")
                elif verbose:
                    print(f"[imp] {model} [{tag}]/{base}: plot failed")

    if not any_plot:
        print("[imp] NO importance plots were produced. Check that each "
              "fold_*/ contains feature_importance_*.csv files with a 'Feature' "
              "column (and 'Dataset' for the long-format scores file).")
        return

    # TDA-vs-no-TDA overlay comparison plots (gradient + perturbation)
    plot_importance_compare_all(verbose=verbose)

    # Across-backbone, scale-free rank comparison (TDA vs no-TDA), aggregating
    # all backbones per covariate instead of illustrating with a single one.
    plot_importance_rank_compare_all(verbose=verbose)


def _wide_cv_path(model: str, tag: str) -> str:
    return os.path.join(_model_dir(model),
                        f"feature_importance_gradients_and_perturbations_{tag}_cv.csv")


def _scores_cv_path(model: str, tag: str) -> str:
    """Per-tree fold-mean CSV for the LONG-format permutation-importance file
    (feature_importance_scores.csv -> _mean_importance_long output). Columns:
    Feature, Dataset, Importance_mean, Importance_std, n_folds."""
    return os.path.join(_model_dir(model),
                        f"feature_importance_scores_{tag}_cv.csv")


def _scores_series(df, split: str):
    """Return (mean Series indexed by Feature, std Series or None) for the
    long-format permutation-importance *_cv.csv, for one split ('train'/'test').
    The value column is the sole '*_mean' column (name-agnostic, so it works
    whether it is 'Importance_mean' or something else)."""
    split = split.lower()
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if not mean_cols or "Dataset" not in df.columns:
        return None, None
    mcol = mean_cols[0]
    scol = mcol[:-5] + "_std"
    sub = df[df["Dataset"].str.lower() == split]
    if sub.empty:
        return None, None
    m = sub.set_index("Feature")[mcol]
    s = sub.set_index("Feature")[scol] if scol in sub.columns else None
    return m, s


def _metric_series(df, metric: str, split: str):
    """Return (mean Series indexed by Feature, std Series or None) for a given
    metric ('gradient'/'perturbation') and split ('train'/'test'), handling BOTH
    schemas:
      (A) wide: column like 'Gradient Importance (Train)_mean' (split in name,
          no Dataset column);
      (B) hybrid: column 'Gradient Importance_mean' with a 'Dataset' column whose
          rows carry the split.
    Returns (None, None) if not found."""
    metric = metric.lower(); split = split.lower()
    has_ds = "Dataset" in df.columns
    for c in df.columns:
        if not c.endswith("_mean"):
            continue
        cl = c.lower()
        if metric not in cl:
            continue
        scol = c[:-5] + "_std"
        if has_ds and split not in cl:
            # schema B: split lives in the Dataset column
            sub = df[df["Dataset"].str.lower() == split]
            if sub.empty:
                continue
            m = sub.set_index("Feature")[c]
            s = sub.set_index("Feature")[scol] if scol in sub.columns else None
            return m, s
        if (not has_ds) and split in cl:
            # schema A: split baked into the column name
            m = df.set_index("Feature")[c]
            s = df.set_index("Feature")[scol] if scol in df.columns else None
            return m, s
    return None, None


def _plot_importance_compare(model: str, metric: str, out_png: str,
                             verbose: bool = True) -> bool:
    """Overlay TDA vs no-TDA fold-mean importance for one metric
    ('gradient', 'perturbation', or 'permutation'), with Train and Test
    side-by-side panels and grouped bars (TDA vs no-TDA) plus std error bars.
    For gradient/perturbation this reads the per-tree wide gradients-and-
    perturbations *_cv.csv; for permutation it reads the per-tree long-format
    feature_importance_scores_{tag}_cv.csv. Both are written by
    plot_importance_means; the reader is tolerant of the differing schemas."""
    is_perm = metric.lower() == "permutation"
    path_fn = _scores_cv_path if is_perm else _wide_cv_path
    paths = {tag: path_fn(model, tag) for tag in ("tda", "notda")}
    dfs = {}
    for tag, p in paths.items():
        if os.path.exists(p):
            dfs[tag] = pd.read_csv(p)
    if "tda" not in dfs and "notda" not in dfs:
        return False

    splits = ["Train", "Test"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), squeeze=False)
    colors = {"tda": "#4C72B0", "notda": "#C44E52"}
    labels = {"tda": "TDA", "notda": "no-TDA"}
    plotted_any = False

    for ax, split in zip(axes[0], splits):
        series = {}
        for tag, df in dfs.items():
            if is_perm:
                m, s = _scores_series(df, split)
            else:
                m, s = _metric_series(df, metric, split)
            if m is None:
                if verbose:
                    haves = [c for c in df.columns if c.endswith("_mean")]
                    print(f"[imp-cmp] {model} {metric}/{split}: no match in "
                          f"[{tag}] (mean cols={haves})")
                continue
            series[tag] = (m, s)
        if not series:
            ax.set_visible(False)
            continue

        ref = next(iter(series.values()))[0]
        feats = list(ref.sort_values().index)
        y = np.arange(len(feats))
        tags = [t for t in ("tda", "notda") if t in series]
        h = 0.8 / len(tags)
        for j, tag in enumerate(tags):
            m, s = series[tag]
            m = m.reindex(feats)
            xerr = s.reindex(feats).fillna(0.0).values if s is not None else None
            ax.barh(y + (j - (len(tags) - 1) / 2) * h, m.values, height=h,
                    xerr=xerr, color=colors[tag], ecolor="#333333", capsize=2,
                    alpha=0.9, label=labels[tag])
        ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0.0, color="#888888", lw=0.8)
        ax.set_title(f"{model} — {metric.capitalize()} ({split})", fontsize=10)
        ax.set_xlabel(f"{metric.capitalize()} importance (fold mean $\\pm$ sd)",
                      fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_importance_compare_all(verbose: bool = True) -> None:
    """For every backbone, write TDA-vs-no-TDA overlay plots: one for gradient
    importance, one for perturbation importance, and one for permutation
    importance (each Train/Test side-by-side)."""
    for model in TSAI_MODELS:
        out_mdir = _model_dir(model)
        if not out_mdir.exists():
            continue
        for metric in ("gradient", "perturbation", "permutation"):
            png = os.path.join(out_mdir,
                               f"feature_importance_{metric}_tda_vs_notda.png")
            ok = _plot_importance_compare(model, metric, png, verbose=verbose)
            if ok and verbose:
                print(f"[imp-cmp] {model}: wrote {os.path.basename(png)}")


# ---------------- across-backbone rank aggregation ----------------

def _read_importance_for_rank(model: str, tag: str, metric: str, split: str):
    """Return a Series indexed by Feature giving the fold-mean importance for one
    backbone/tag/metric/split, from the per-backbone *_cv.csv files. `metric` is
    'gradient'|'perturbation'|'permutation'; 'split' is 'Train'|'Test'. Returns
    None if the file/column is absent.

    The gradient/perturbation CV file appears in two layouts across trees:
      (a) wide-by-split: columns like 'Gradient Importance (Train)_mean';
      (b) long-by-Dataset: a 'Dataset' column plus 'Gradient Importance_mean'.
    Both are handled so TDA and no-TDA are read symmetrically."""
    mdir = _model_dir(model)
    if metric == "permutation":
        p = os.path.join(mdir, f"feature_importance_scores_{tag}_cv.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        if "Dataset" not in df.columns or "Importance_mean" not in df.columns:
            return None
        sub = df[df["Dataset"].str.lower() == split.lower()]
        if sub.empty:
            return None
        return sub.set_index("Feature")["Importance_mean"]

    # gradient / perturbation
    p = os.path.join(mdir,
                     f"feature_importance_gradients_and_perturbations_{tag}_cv.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "Feature" not in df.columns:
        return None
    label = "Gradient" if metric == "gradient" else "Perturbation"

    # layout (a): split encoded in the column name
    col_wide = f"{label} Importance ({split})_mean"
    if col_wide in df.columns:
        return df.set_index("Feature")[col_wide]

    # layout (b): split encoded in a Dataset column, plain value column
    col_long = f"{label} Importance_mean"
    if "Dataset" in df.columns and col_long in df.columns:
        sub = df[df["Dataset"].str.lower() == split.lower()]
        if sub.empty:
            return None
        return sub.set_index("Feature")[col_long]

    return None


def _rank_within_backbone(s):
    """Rank features so that rank 1 = MOST important. Ties get average ranks.
    Returns a Series of ranks indexed by Feature."""
    # higher importance -> smaller (better) rank number
    return s.rank(ascending=False, method="average")


def plot_importance_rank_compare(metric: str, split: str, out_png: str,
                                 verbose: bool = True) -> bool:
    """Across-backbone TDA-vs-no-TDA comparison of covariate importance, made
    scale-free by ranking within each backbone before aggregating.

    For each backbone the covariates are ranked (1 = most important) under the
    given metric and split, separately for the TDA and no-TDA models; the ranks
    are then aggregated across backbones per covariate. The plot shows, per
    covariate, the mean rank (marker) with a min-max whisker and a +/-1 SD band
    across backbones, TDA (blue) vs no-TDA (red). Covariates are ordered by the
    TDA mean rank. Lower rank = more important, so the y-axis is inverted.

    Ranking removes the cross-architecture scale differences that make raw
    importance magnitudes non-comparable (gradient magnitudes in particular vary
    by orders of magnitude between backbones)."""
    tags = ("tda", "notda")
    # collect per-backbone rank vectors
    ranks = {tag: [] for tag in tags}
    feats_union = None
    for model in TSAI_MODELS:
        for tag in tags:
            s = _read_importance_for_rank(model, tag, metric, split)
            if s is None or s.empty:
                continue
            r = _rank_within_backbone(s)
            ranks[tag].append(r)
            feats_union = r.index if feats_union is None else feats_union.union(r.index)
    if feats_union is None or (not ranks["tda"] and not ranks["notda"]):
        return False

    def stack(tag):
        if not ranks[tag]:
            return None
        M = pd.concat(ranks[tag], axis=1).reindex(feats_union)  # feats x backbones
        return M

    Mt, Mn = stack("tda"), stack("notda")

    # order covariates by TDA mean rank (fallback to notda)
    order_src = Mt if Mt is not None else Mn
    order = order_src.mean(axis=1).sort_values().index.tolist()

    y = np.arange(len(order))[::-1]  # top of plot = most important
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(order) + 1.5))
    colors = {"tda": "#4C72B0", "notda": "#C44E52"}
    labels = {"tda": "TDA", "notda": "no-TDA"}
    offset = {"tda": +0.16, "notda": -0.16}

    color_handles = []
    for tag, M in (("tda", Mt), ("notda", Mn)):
        if M is None:
            continue
        m = M.reindex(order)
        mean = m.mean(axis=1).to_numpy()
        sd = m.std(axis=1).to_numpy()
        lo = m.min(axis=1).to_numpy()
        hi = m.max(axis=1).to_numpy()
        yy = y + offset[tag]
        # min-max whisker
        ax.hlines(yy, lo, hi, color=colors[tag], alpha=0.45, lw=1.2, zorder=1)
        # +/-1 SD box as a thick segment
        ax.hlines(yy, mean - sd, mean + sd, color=colors[tag], alpha=0.9,
                  lw=4.0, zorder=2)
        # mean marker
        h, = ax.plot(mean, yy, "o", color=colors[tag], ms=6, zorder=3,
                     label=labels[tag])
        color_handles.append(h)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Importance rank across backbones (1 = most important)")
    ax.set_title(f"{metric.capitalize()} importance rank - TDA vs no-TDA ({split})")
    ax.invert_xaxis()  # rank 1 (most important) on the right-> put it on left

    # Two-part legend: color key (TDA / no-TDA) + an encoding key explaining the
    # marker, thick bar, and thin whisker, drawn as neutral-grey glyphs.
    
    enc_handles = [
        Line2D([0], [0], color="#666666", marker="o", lw=0, ms=6,
               label="mean rank"),
        Line2D([0], [0], color="#666666", lw=4.0, label=r"$\pm$1 SD"),
        Line2D([0], [0], color="#666666", lw=1.2, alpha=0.6,
               label="min-max"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="lower right", fontsize=9,
                     title="model")
    ax.add_artist(leg1)
    ax.legend(handles=enc_handles, loc="upper left", fontsize=8,
              title="across backbones", framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def plot_importance_rank_compare_all(verbose: bool = True) -> None:
    """Write across-backbone rank-comparison plots (TDA vs no-TDA) for each
    importance metric and split, into OUT_ROOT."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for metric in ("gradient", "perturbation", "permutation"):
        for split in ("Train", "Test"):
            png = os.path.join(str(OUT_ROOT),
                               f"importance_rank_{metric}_{split.lower()}_tda_vs_notda.png")
            ok = plot_importance_rank_compare(metric, split, png, verbose=verbose)
            if verbose:
                msg = "wrote" if ok else "skipped (no data)"
                print(f"[imp-rank] {metric}/{split}: {msg} {os.path.basename(png)}")


def run_engine() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    y = load_y()
    folds = make_folds(y)
    print(f"[folds] built {len(folds)} folds (seed={SEED}); "
          f"test sizes: {[len(te) for _, _, te in folds]}")

    backbone_rows = []
    for model in TSAI_MODELS:
        n_done = 0
        for k, fsplit in enumerate(folds):
            if _run_one_fold(model, k, fsplit) is not None:
                n_done += 1
        if n_done == 0:
            print(f"[warn] {model}: no folds completed")
            continue
        if _aggregate_model_folds(model):
            print(f"[ok] {model}: {n_done} folds compared + aggregated")
        try:
            s = pd.read_csv(_model_dir(model) / "summary.csv")
            row = {"model": model, "n_folds": n_done}
            for c in s.select_dtypes(include=[np.number]).columns:
                row[c] = float(s[c].iloc[0])
            backbone_rows.append(row)
        except Exception as e:
            print(f"[warn] {model}: summary read failed: {e}")

    if backbone_rows:
        agg = pd.DataFrame(backbone_rows)
        agg_path = OUT_ROOT / f"aggregate_cv_n{NEPOCHS}.csv"
        agg.to_csv(agg_path, index=False)
        print(f"[aggregate] wrote {agg_path}")

    try:
        print("[plots] generating feature-importance fold-mean plots ...")
        plot_importance_means()
    except Exception as e:
        print(f"[plots] importance plots FAILED: {e}")

    try:
        print("[plots] generating summary + reliability + CPD plots ...")
        cfg = PlotConfig(nepochs=NEPOCHS, split=SPLIT, _summary_root=OUT_ROOT)
        ReportRunner(cfg).run()
        print("[plots] done.")
    except Exception as e:
        print(f"[plots] FAILED: {e}")
        
        traceback.print_exc()


# ============================ POOLED ROUTE ============================

def _pooled_oof(backbone_folder):
    """Concatenate all folds' held-out TEST predictions for one backbone."""
    files = sorted(glob.glob(os.path.join(backbone_folder, "fold_*", "fold_predictions.npz")))
    if not files:
        return None, None
    P, Y = [], []
    for f in files:
        d = np.load(f)
        P.append(d["test_probs"])
        Y.append(d["test_targets"])
    return np.concatenate(P, 0), np.concatenate(Y, 0)


# --------------------------- reliability curves ---------------------------

def _macro_reliability_curve(y, p, nc, n_bins=10):
    """Macro one-vs-rest reliability curve for a pooled (prob, label) set.

    Mirrors the macro-OvR ECE construction: for each class the class-j
    probability is binned into `n_bins` equal-width bins on [0,1]; within each
    bin we record mean predicted probability (confidence) and empirical positive
    rate (accuracy); the two are averaged across classes bin-by-bin (over the
    classes that populate a given bin), giving a single macro curve. Also returns
    the macro-OvR ECE implied by the same binning, so the curve and the ECE
    legend are guaranteed mutually consistent.

    Returns (conf[n_bins], acc[n_bins], ece) with NaNs in unpopulated bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    yb = label_binarize(y, classes=list(range(nc)))
    if yb.ndim == 1:  # binary edge case
        yb = np.column_stack([1 - yb, yb])
    conf_bc = np.full((nc, n_bins), np.nan)
    acc_bc  = np.full((nc, n_bins), np.nan)
    w_bc    = np.zeros((nc, n_bins))
    n = len(y)
    for j in range(nc):
        s = p[:, j]
        b = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
        for k in range(n_bins):
            m = (b == k)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            conf_bc[j, k] = s[m].mean()
            acc_bc[j, k]  = yb[m, j].mean()
            w_bc[j, k]    = cnt / n
    # macro average over classes populating each bin
    with np.errstate(invalid="ignore"):
        conf = np.nanmean(conf_bc, axis=0)
        acc  = np.nanmean(acc_bc,  axis=0)
    # macro-OvR ECE from the same bins: per-class sum_b w_b|acc-conf|, then mean
    per_class_ece = np.nansum(w_bc * np.abs(acc_bc - conf_bc), axis=1)
    ece = float(np.mean(per_class_ece)) if nc else float("nan")
    return conf, acc, ece


def _reliability_overlay(items, title, out_png, n_bins=10):
    """Draw one reliability figure overlaying several backbones.

    `items` is a list of (label, y, p, nc). Each backbone contributes one macro
    reliability curve; the perfect-calibration diagonal is drawn for reference,
    and each curve's macro-OvR ECE is shown in the legend. A pooled curve over
    all concatenated predictions is added as a thick black line.
    """
    if not items:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.3, color="#888888",
            label="perfect calibration", zorder=1)
    cmap = plt.cm.get_cmap("tab10")
    Pall, Yall, nc0 = [], [], None
    for i, (label, y, p, nc) in enumerate(sorted(items, key=lambda t: t[0].lower())):
        nc0 = nc
        conf, acc, ece = _macro_reliability_curve(y, p, nc, n_bins=n_bins)
        ok = ~np.isnan(conf) & ~np.isnan(acc)
        ax.plot(conf[ok], acc[ok], marker="o", ms=3.5, lw=1.4,
                color=cmap(i % 10), alpha=0.85,
                label=f"{label} (ECE {ece:.3f})", zorder=2)
        Pall.append(p); Yall.append(y)
    # pooled curve
    if nc0 is not None:
        yP = np.concatenate(Yall, 0); pP = np.concatenate(Pall, 0)
        conf, acc, ece = _macro_reliability_curve(yP, pP, nc0, n_bins=n_bins)
        ok = ~np.isnan(conf) & ~np.isnan(acc)
        ax.plot(conf[ok], acc[ok], marker="s", ms=5, lw=2.6, color="black",
                label=f"pooled (ECE {ece:.3f})", zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability (confidence)")
    ax.set_ylabel("empirical fraction of positives (accuracy)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def run_reliability(base_root: str, tda_root: str, out_dir: str, nc: int,
                    n_bins: int = 10) -> None:
    """Emit three macro-OvR reliability figures: all TDA backbones overlaid, all
    base backbones overlaid, and (for continuity) the existing single pooled
    TDA-vs-base curve. Reuses the same pooled out-of-fold predictions as the
    `pooled` route, so every curve is consistent with the reported ECE tables."""
    os.makedirs(out_dir, exist_ok=True)

    def backbones(root):
        return {_clean_name(d): d for d in glob.glob(os.path.join(root, "*_val_*"))
                if os.path.isdir(d)}

    B, T = backbones(base_root), backbones(tda_root)
    common = sorted(set(B) & set(T))
    print(f"[reliability] {len(common)} backbones present in both trees")

    tda_items, base_items = [], []
    pooled_pt, pooled_yt, pooled_pb, pooled_yb = [], [], [], []
    for name in common:
        pt, yt = _pooled_oof(T[name])
        pb, yb = _pooled_oof(B[name])
        if pt is not None:
            tda_items.append((name, yt, pt, nc)); pooled_pt.append(pt); pooled_yt.append(yt)
        if pb is not None:
            base_items.append((name, yb, pb, nc)); pooled_pb.append(pb); pooled_yb.append(yb)

    p1 = os.path.join(out_dir, "reliability_tda_all_backbones.png")
    p2 = os.path.join(out_dir, "reliability_base_all_backbones.png")
    if _reliability_overlay(tda_items, "Macro-OvR reliability — TDA backbones", p1, n_bins):
        print(f"[reliability] wrote {p1}")
    if _reliability_overlay(base_items, "Macro-OvR reliability — base backbones", p2, n_bins):
        print(f"[reliability] wrote {p2}")

    # pooled TDA-vs-base single-curve figure (continuity with figure_3_11)
    if pooled_pt and pooled_pb:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.plot([0, 1], [0, 1], ls="--", lw=1.3, color="#888888",
                label="perfect calibration")
        for lab, Ps, Ys, col in [("base", pooled_pb, pooled_yb, "#C44E52"),
                                 ("TDA",  pooled_pt, pooled_yt, "#4C72B0")]:
            yP = np.concatenate(Ys, 0); pP = np.concatenate(Ps, 0)
            conf, acc, ece = _macro_reliability_curve(yP, pP, nc, n_bins=n_bins)
            ok = ~np.isnan(conf) & ~np.isnan(acc)
            ax.plot(conf[ok], acc[ok], marker="o", ms=4, lw=2.0, color=col,
                    label=f"{lab} (ECE {ece:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", "box")
        ax.set_xlabel("mean predicted probability (confidence)")
        ax.set_ylabel("empirical fraction of positives (accuracy)")
        ax.set_title("Macro-OvR reliability — pooled over all backbones")
        ax.legend(fontsize=9, loc="upper left")
        fig.tight_layout()
        pp = os.path.join(out_dir, "reliability_pooled_tda_vs_base.png")
        fig.savefig(pp, dpi=200, bbox_inches="tight")
        fig.savefig(os.path.splitext(pp)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[reliability] wrote {pp}")


def _macro_auc(y, p, nc):
    yb = label_binarize(y, classes=list(range(nc)))
    cols = [i for i in range(nc) if 0 < yb[:, i].sum() < len(y)]
    if not cols:
        return np.nan
    return roc_auc_score(yb[:, cols], p[:, cols], average="macro", multi_class="ovr")


def _mcnemar(y, pred_b, pred_t):
    bc = (pred_b == y); tc = (pred_t == y)
    n01 = int(np.sum(~bc & tc))   # bare wrong, tda right (tda gains)
    n10 = int(np.sum(bc & ~tc))   # bare right, tda wrong (tda loses)
    n = n01 + n10
    p = binomtest(min(n01, n10), n, 0.5).pvalue if n > 0 else 1.0
    return n01, n10, float(p)


def _auc_delta_boot(y, pb, pt, nc, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = _macro_auc(yy, pb[idx], nc); b = _macro_auc(yy, pt[idx], nc)
        if not (np.isnan(a) or np.isnan(b)):
            d.append(b - a)
    d = np.array(d)
    delta = _macro_auc(y, pt, nc) - _macro_auc(y, pb, nc)
    lo = float(np.percentile(d, 2.5)) if len(d) else np.nan
    hi = float(np.percentile(d, 97.5)) if len(d) else np.nan
    return float(delta), lo, hi


def _clean_name(folder):
    base = os.path.basename(folder.rstrip("/"))
    return base.split("_val_")[0] if "_val_" in base else base


def compare_backbone(name, base_folder, tda_folder, nc, n_boot=2000):
    pb, yb = _pooled_oof(base_folder)
    pt, yt = _pooled_oof(tda_folder)
    if pb is None or pt is None:
        print(f"  [skip] {name}: missing fold_predictions.npz in one pipeline")
        return None
    # paired folds => same countries/labels in both; verify order matches
    if not np.array_equal(yb, yt):
        print(f"  [warn] {name}: pooled label vectors differ between pipelines "
              f"(bare n={len(yb)}, tda n={len(yt)}). Folds may not be paired; "
              f"comparison treats them as independent pooled sets.")
    y = yb
    predb, predt = pb.argmax(1), pt.argmax(1)
    n01, n10, mp = _mcnemar(y, predb, predt)
    delta, lo, hi = _auc_delta_boot(y, pb, pt, nc, n_boot=n_boot)
    row = {
        "backbone": name, "n": int(len(y)),
        "acc_base": float((predb == y).mean()), "acc_tda": float((predt == y).mean()),
        "kappa_base": float(cohen_kappa_score(y, predb, weights="quadratic", labels=list(range(nc)))),
        "kappa_tda":  float(cohen_kappa_score(y, predt, weights="quadratic", labels=list(range(nc)))),
        "macro_auc_base": _macro_auc(y, pb, nc), "macro_auc_tda": _macro_auc(y, pt, nc),
        "auc_delta": delta, "auc_delta_lo": lo, "auc_delta_hi": hi,
        "mcnemar_tda_gains": n01, "mcnemar_tda_loses": n10, "mcnemar_p": mp,
    }
    sig = "*" if (not np.isnan(lo) and (lo > 0 or hi < 0)) else " "
    print(f"  {name:18s} AUC base={row['macro_auc_base']:.3f} tda={row['macro_auc_tda']:.3f} "
          f"delta={delta:+.3f}[{lo:+.3f},{hi:+.3f}]{sig}  "
          f"kappa {row['kappa_base']:.3f}->{row['kappa_tda']:.3f}  "
          f"McNemar p={mp:.3f} (+{n01}/-{n10})")
    return row


def _write_pooled_latex(df: pd.DataFrame, out_dir: str) -> str:
    def esc(s):
        return str(s).replace("_", r"\_")
    L = [r"% Auto-generated by compare_all.py (pooled route)",
         r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Paired bare-vs-TDA comparison on pooled out-of-fold "
         r"predictions (each country held out once; identical folds in both "
         r"pipelines). $\Delta$AUC is macro-AUC(TDA)$-$macro-AUC(bare) with a "
         r"paired bootstrap 95\% CI; McNemar tests paired hard-prediction "
         r"agreement. $\kappa$ is quadratic-weighted.}",
         r"\label{tab:tda_vs_bare}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Backbone & AUC bare & AUC TDA & $\Delta$AUC [95\% CI] & McNemar $p$ \\",
         r"\midrule"]
    for _, r in df.iterrows():
        ci = f"[{r['auc_delta_lo']:+.3f}, {r['auc_delta_hi']:+.3f}]"
        L.append(f"{esc(r['backbone'])} & {r['macro_auc_base']:.3f} & "
                 f"{r['macro_auc_tda']:.3f} & {r['auc_delta']:+.3f} {ci} & "
                 f"{r['mcnemar_p']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(out_dir, "tda_vs_bare_all_backbones.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return tex_path


def run_pooled(base_root: str, tda_root: str, out_dir: str, nc: int) -> None:
    print(f"  bare: {base_root}")
    print(f"  tda : {tda_root}")
    print(f"  out : {out_dir}\n")
    os.makedirs(out_dir, exist_ok=True)

    def backbones(root):
        return {_clean_name(d): d for d in glob.glob(os.path.join(root, "*_val_*"))
                if os.path.isdir(d)}

    B, T = backbones(base_root), backbones(tda_root)
    common = sorted(set(B) & set(T))
    print(f"comparing {len(common)} backbones present in both trees\n")

    rows = []
    for name in common:
        r = compare_backbone(name, B[name], T[name], nc)
        if r:
            rows.append(r)

    if not rows:
        print("no comparable backbones")
        return
    df = pd.DataFrame(rows).sort_values("auc_delta", ascending=False)
    csv_path = os.path.join(out_dir, "tda_vs_bare_all_backbones.csv")
    df.to_csv(csv_path, index=False)
    tex_path = _write_pooled_latex(df, out_dir)

    print(f"\nwrote {csv_path}")
    print(f"wrote {tex_path}")
    pos = (df["auc_delta"] > 0).sum()
    print(f"\nTDA improved macro-AUC in {pos}/{len(df)} backbones; "
          f"mean delta = {df['auc_delta'].mean():+.3f}")


# ============================ LATEX TABLES ============================
# All three paper tables are built from the SAME pair of per-model summary CSVs
# (TDA and no-TDA), so shared cells cannot disagree across tables.

def _load_summary(path) -> pd.DataFrame:
    """Load an all_models_performance_summary CSV; index by model_name."""
    df = pd.read_csv(path)
    if "model_name" not in df.columns:
        raise ValueError(f"{path}: expected a 'model_name' column "
                         f"(got {list(df.columns)})")
    return df.set_index("model_name")


def _cell(mean, std, bold=False):
    v = f"{float(mean):.3f}$\\pm${float(std):.3f}"
    return f"\\textbf{{{v}}}" if bold else v


def _perf_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """One performance table (Train/Valid/Test accuracy and AUROC), rows ordered
    by test AUROC, best test value per column in bold. Built from a single
    summary frame so it always matches its source."""
    d = df.reset_index().sort_values("test_auc", ascending=False)
    best_ta = d["test_accuracy"].max()
    best_tauc = d["test_auc"].max()
    lines = []
    n = len(d)
    for i, (_, r) in enumerate(d.iterrows()):
        name = str(r["model_name"]).replace("_", r"\_")
        ta = _cell(r["train_accuracy"], r["train_accuracy_std"])
        va = _cell(r["valid_accuracy"], r["valid_accuracy_std"])
        te = _cell(r["test_accuracy"], r["test_accuracy_std"],
                   r["test_accuracy"] == best_ta)
        tr = _cell(r["train_auc"], r["train_auc_std"])
        vr = _cell(r["valid_auc"], r["valid_auc_std"])
        ter = _cell(r["test_auc"], r["test_auc_std"], r["test_auc"] == best_tauc)
        lines.append(f"    {name} & {ta} & {va} & {te} & {tr} & {vr} & {ter} \\\\")
        lines.append("     \\hline" if i < n - 1 else "     \\hline \\hline")
    body = "\n".join(lines)
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\begin{tabular}{l | ccc | ccc}\n"
        "    \\hline\\hline\n"
        "   \\textbf{Model}  & \\multicolumn{3}{c|}{\\textbf{Accuracy}} & "
        "\\multicolumn{3}{c}{\\textbf{AUROC}} \\\\\n"
        "    \\cline{2-4}  \\cline{5-7}\n"
        "     & \\textbf{Train} & \\textbf{Valid} & \\textbf{Test} & "
        "\\textbf{Train} & \\textbf{Valid} & \\textbf{Test} \\\\\n"
        "    \\hline\\hline\n"
        f"{body}\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def _overfitting_table(tda: pd.DataFrame, base: pd.DataFrame,
                       caption_label="tab:overfitting") -> str:
    """Paired Base-vs-TDA train/test accuracy overfitting table, rows ordered by
    Delta-Test descending, with a Spearman(Delta-Train, Delta-Test) in the
    caption. Built from the SAME two summary frames as the performance tables."""
    models = [m for m in tda.index if m in base.index]
    rows = []
    for m in models:
        rows.append({
            "model": m,
            "train_base": float(base.loc[m, "train_accuracy"]),
            "train_tda":  float(tda.loc[m, "train_accuracy"]),
            "test_base":  float(base.loc[m, "test_accuracy"]),
            "test_tda":   float(tda.loc[m, "test_accuracy"]),
        })
    d = pd.DataFrame(rows)
    d["dtrain"] = d["train_tda"] - d["train_base"]
    d["dtest"] = d["test_tda"] - d["test_base"]
    d = d.sort_values("dtest", ascending=False)

    # Spearman rho between dTrain and dTest (no SciPy dependency: rank + Pearson)
    rt = d["dtrain"].rank().to_numpy()
    re = d["dtest"].rank().to_numpy()
    if len(d) >= 3 and rt.std() > 0 and re.std() > 0:
        rho = float(np.corrcoef(rt, re)[0, 1])
        # two-sided p via t-approximation
        nobs = len(d)
        if abs(rho) < 1.0:
            tval = rho * np.sqrt((nobs - 2) / (1 - rho ** 2))
            # survival of |t| under t_{n-2}, math.erf-free normal approx fallback
            try:
                pval = float(2 * _tdist.sf(abs(tval), nobs - 2))
            except Exception:
                pval = float("nan")
        else:
            pval = 0.0
    else:
        rho, pval = float("nan"), float("nan")
    rho_str = f"$\\rho = {rho:.3f}$" if rho == rho else "$\\rho = $ n/a"
    p_str = (f"($p = {pval:.3f}$)" if pval == pval else "")

    caption = ("Training and test accuracy under base and TDA models. All values "
               "are means across 5 country-level CV folds. Spearman correlation "
               f"between $\\Delta$Train and $\\Delta$Test: {rho_str} {p_str}.")

    lines = []
    n = len(d)
    for i, (_, r) in enumerate(d.iterrows()):
        name = str(r["model"]).replace("_", r"\_")
        lines.append(
            f"  {name} & {r['train_base']:.3f} & {r['train_tda']:.3f} & "
            f"{r['dtrain']:+.3f} & {r['test_base']:.3f} & {r['test_tda']:.3f} & "
            f"{r['dtest']:+.3f} \\\\")
        if i < n - 1:
            lines.append("  \\hline")
    body = "\n".join(lines)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{caption_label}}}\n"
        "\\small\n"
        "\\begin{tabular}{@{}l | rr r | rr r@{}}\n"
        "\\hline\\hline\n"
        "\\textbf{Backbone} & \\multicolumn{3}{c|}{\\textbf{Training Accuracy}} & "
        "\\multicolumn{3}{c}{\\textbf{Test Accuracy}} \\\\\n"
        "\\cline{2-7}\n"
        "\\textbf{Module}  & Base & TDA & $\\Delta$ & Base & TDA & $\\Delta$ \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _plot_overfitting_scatter(tda: pd.DataFrame, base: pd.DataFrame,
                              out_png: str, verbose: bool = True) -> bool:
    """Scatter of Delta-train vs Delta-test (TDA - Base) per backbone, with a
    regression line and Spearman rho in the title, for both Accuracy and AUC.
    Built from the SAME summary pair as the overfitting table."""
  
    models = [m for m in tda.index if m in base.index]
    if len(models) < 3:
        if verbose:
            print(f"[tables] overfitting scatter: too few shared backbones "
                  f"({len(models)}) — skip")
        return False
    d = pd.DataFrame({
        "backbone": models,
        "dtr_acc": [float(tda.loc[m, "train_accuracy"] - base.loc[m, "train_accuracy"]) for m in models],
        "dte_acc": [float(tda.loc[m, "test_accuracy"] - base.loc[m, "test_accuracy"]) for m in models],
        "dtr_auc": [float(tda.loc[m, "train_auc"] - base.loc[m, "train_auc"]) for m in models],
        "dte_auc": [float(tda.loc[m, "test_auc"] - base.loc[m, "test_auc"]) for m in models],
    })

    rho_acc, p_acc = _st.spearmanr(d["dtr_acc"], d["dte_acc"])
    rho_auc, p_auc = _st.spearmanr(d["dtr_auc"], d["dte_auc"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (xc, yc, rho, pval, lab) in zip(
            axes,
            [("dtr_acc", "dte_acc", rho_acc, p_acc, "Accuracy"),
             ("dtr_auc", "dte_auc", rho_auc, p_auc, "AUC")]):
        ax.scatter(d[xc], d[yc], s=60, zorder=5, color="#4C72B0")
        for _, r in d.iterrows():
            ax.annotate(r["backbone"], (r[xc], r[yc]), fontsize=7,
                        ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")
        if d[xc].std() > 0:
            slope, intercept, *_ = _st.linregress(d[xc], d[yc])
            xs = np.array([d[xc].min(), d[xc].max()])
            ax.plot(xs, intercept + slope * xs, "r--", lw=1.5, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
        ax.set_xlabel(f"$\\Delta$ Training {lab} (TDA $-$ Base)")
        ax.set_ylabel(f"$\\Delta$ Test {lab} (TDA $-$ Base)")
        ax.set_title(f"{lab}: Spearman $\\rho$ = {rho:.3f} (p = {pval:.3f})")
        ax.grid(alpha=0.3)
    fig.suptitle("Does TDA-induced overfitting correlate with generalization gains?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def run_tables(tda_csv: str, base_csv: str, out_dir: str) -> None:
    """Emit all three LaTeX tables from the single (TDA, no-TDA) summary pair."""
    os.makedirs(out_dir, exist_ok=True)
    for label, p in [("TDA", tda_csv), ("no-TDA", base_csv)]:
        if not os.path.exists(p):
            print(f"[tables] MISSING {label} summary: {p}")
    tda = _load_summary(tda_csv)
    base = _load_summary(base_csv)

    t1 = _perf_table(
        tda,
        caption=("Cross-validated classification performance of ten "
                 "\\textit{tsai} backbones under the \\textbf{TDA-conditioned} "
                 "classifier. All models were trained for 900 epochs and "
                 "evaluated with 5-fold stratified cross-validation (validation "
                 "split $0.2$, test split $0.1$); reported values are "
                 "mean${}\\pm{}$standard deviation across folds. Accuracy is "
                 "multiclass classification accuracy and AUROC is the macro "
                 "one-vs-rest area under the ROC curve. Rows are ordered by test "
                 "AUROC; the best test-set value in each column is shown in bold."),
        label="tab:model-performance")
    t2 = _perf_table(
        base,
        caption=("Cross-validated classification performance of ten "
                 "\\textit{tsai} backbones under the base classifier "
                 "\\emph{without} topological conditioning (\\textbf{no-TDA "
                 "baseline}). All models were trained for 900 epochs and "
                 "evaluated with 5-fold stratified cross-validation (validation "
                 "split $0.2$, test split $0.1$); reported values are "
                 "mean${}\\pm{}$standard deviation across folds. Accuracy is "
                 "multiclass classification accuracy and AUROC is the macro "
                 "one-vs-rest area under the ROC curve. Rows are ordered by test "
                 "AUROC; the best test-set value in each column is shown in bold."),
        label="tab:model-performance-notda")
    t3 = _overfitting_table(tda, base)

    out = {
        "model_performance_table.tex": t1,
        "model_performance_table_notda.tex": t2,
        "overfitting_table.tex": t3,
    }
    for fname, txt in out.items():
        path = os.path.join(out_dir, fname)
        with open(path, "w") as fh:
            fh.write(txt)
        print(f"[tables] wrote {path}")

    # overfitting scatter plot (Delta-train vs Delta-test), same summary pair
    scatter_png = os.path.join(out_dir, "overfitting_vs_generalization.png")
    if _plot_overfitting_scatter(tda, base, scatter_png):
        print(f"[tables] wrote {scatter_png}")

    print("[tables] all tables + overfitting plot built from the same summary "
          "pair (mutually consistent).")


# ============================ POOLED WILCOXON ============================
# Headline statistical test: pool fold-level TDA-Base deltas across ALL
# backbones (10 backbones x 5 folds = 50 obs/metric) and run one Wilcoxon
# signed-rank test per metric, plus an across-backbone sign test. Reads the
# per-fold comparison summaries the engine route already writes
# (OUT_ROOT/<backbone>/per_fold_summary.csv with columns metric, delta,
# higher_is_better). 

def _wilcoxon_metric_order():
    return ["Accuracy", "F1_macro", "F1_micro", "BalancedAccuracy",
            "AUROC_macro", "AUPRC_macroOvR", "ECE_macroOvR", "WrongPredConfidence"]


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _pooled_wilcoxon_table(result_ordered: pd.DataFrame) -> str:
    lines = []
    n = len(result_ordered)
    for i, (_, r) in enumerate(result_ordered.iterrows()):
        m = str(r["metric"]).replace("_", r"\_")
        hib = bool(r["higher_is_better"])
        dir_arrow = "$\\uparrow$" if hib else "$\\downarrow$"
        mean_d = f"{r['mean_delta']:+.4f}"
        p_val = r["pooled_wilcoxon_p"]
        p_str = _fmt_p(p_val) + _stars(p_val)
        sign_str = f"{int(r['n_favorable_backbones'])}/{int(r['n_backbones'])}"
        if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) \
                and p_val < 0.05:
            favorable = (hib and r["mean_delta"] > 0) or (not hib and r["mean_delta"] < 0)
            verdict = "\\textbf{TDA helps}" if favorable else "\\textbf{TDA hurts}"
        else:
            verdict = "n.s."
        lines.append(
            f"    {m} & {dir_arrow} & {mean_d} & {p_str} & {sign_str} & {verdict} \\\\")
        lines.append("     \\hline" if i < n - 1 else "     \\hline \\hline")
    body = "\n".join(lines)
    return (
        "% * p < 0.05, ** p < 0.01, *** p < 0.001 (one-sided)\n"
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Pooled Wilcoxon signed-rank tests on fold-level "
        "TDA$-$Base deltas across all 10 backbone architectures ($n = 50$ "
        "fold-level observations per metric). The Sign column reports the number "
        "of backbone-level means in the favorable direction (out of 10). "
        "$\\uparrow$ = higher is better; $\\downarrow$ = lower is better.}\n"
        "  \\label{tab:pooled-wilcoxon}\n"
        "  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\begin{tabular}{l | c r r r c}\n"
        "    \\hline\\hline\n"
        "    \\textbf{Metric} & \\textbf{Dir} & \\textbf{Mean $\\Delta$} & "
        "\\textbf{Wilcoxon $p$} & \\textbf{Sign} & \\textbf{Verdict} \\\\\n"
        "    \\hline\\hline\n"
        f"{body}\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def run_stats(cv_root: str, out_dir: str) -> None:
    """Pooled Wilcoxon signed-rank + sign tests across all backbones, written as
    a CSV and a LaTeX table in the same style as the other tables."""
   
    cv_root = Path(cv_root)
    os.makedirs(out_dir, exist_ok=True)

    backbones = sorted(d.name for d in cv_root.iterdir()
                       if d.is_dir() and (d / "per_fold_summary.csv").exists())
    if not backbones:
        print(f"[stats] no <backbone>/per_fold_summary.csv under {cv_root} — "
              f"run `engine` first.")
        return
    print(f"[stats] found {len(backbones)} backbones")

    frames = []
    for bb in backbones:
        df = pd.read_csv(cv_root / bb / "per_fold_summary.csv")
        df["backbone"] = bb
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)

    required = {"metric", "delta", "higher_is_better"}
    if not required.issubset(all_data.columns):
        print(f"[stats] per_fold_summary.csv lacks required columns "
              f"{sorted(required)} (have {list(all_data.columns)}); cannot run "
              f"pooled Wilcoxon.")
        return

    n_backbones = len(backbones)
    rows = []
    for metric in sorted(all_data["metric"].unique()):
        sub = all_data[all_data["metric"] == metric]
        deltas = sub["delta"].to_numpy(dtype=float)
        hib = bool(sub["higher_is_better"].iloc[0])
        n = len(deltas)
        alt = "greater" if hib else "less"
        try:
            stat, p_pooled = stats.wilcoxon(deltas, alternative=alt)
        except Exception:
            stat, p_pooled = np.nan, np.nan
        backbone_means = sub.groupby("backbone")["delta"].mean()
        n_fav_bb = int((backbone_means > 0).sum()) if hib \
            else int((backbone_means < 0).sum())
        try:
            sign_p = binomtest(n_fav_bb, n_backbones, 0.5,
                               alternative="greater").pvalue
        except Exception:
            sign_p = np.nan
        n_fav_folds = int((deltas > 0).sum()) if hib else int((deltas < 0).sum())
        if p_pooled is not None and not np.isnan(p_pooled) and p_pooled < 0.05:
            favorable = (hib and deltas.mean() > 0) or (not hib and deltas.mean() < 0)
            direction = "favorable" if favorable else "unfavorable"
        else:
            direction = "n.s."
        rows.append({
            "metric": metric, "higher_is_better": hib,
            "n_pooled": n, "n_backbones": n_backbones,
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if n > 1 else 0.0,
            "median_delta": float(np.median(deltas)),
            "pooled_wilcoxon_stat": None if np.isnan(stat) else float(stat),
            "pooled_wilcoxon_p": None if np.isnan(p_pooled) else float(p_pooled),
            "n_favorable_backbones": n_fav_bb,
            "sign_test_p": None if np.isnan(sign_p) else float(sign_p),
            "n_favorable_folds": n_fav_folds,
            "frac_favorable_folds": n_fav_folds / n if n else 0.0,
            "direction": direction,
        })
    result = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "pooled_wilcoxon_results.csv")
    result.to_csv(csv_path, index=False)
    print(f"[stats] wrote {csv_path}")

    ordered = (result.set_index("metric")
               .reindex(_wilcoxon_metric_order()).reset_index()
               .dropna(subset=["higher_is_better"]))
    if ordered.empty:
        ordered = result  # metrics not in the canonical list: emit as found
    tex_path = os.path.join(out_dir, "pooled_wilcoxon_results.tex")
    with open(tex_path, "w") as fh:
        fh.write(_pooled_wilcoxon_table(ordered))
    print(f"[stats] wrote {tex_path}")
    print("[stats] pooled Wilcoxon + sign tests complete.")


# ============================ CLI ============================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Consolidated bare-vs-TDA comparison pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("engine", help="per-fold full-engine comparison + plots")

    sub.add_parser("importance",
                   help="(re)build feature-importance fold-means + plots only, "
                        "from existing fold_*/ outputs (no engine run)")
    
    sub.add_parser("plots", help="regenerate summary + reliability + CPD plots "
                                  "from existing per-fold CSVs (no engine re-run)")

    pp = sub.add_parser("pooled", help="engine-free pooled paired comparison + table")
    pp.add_argument("--base-root", default=str(BASE_RESULTS_ROOT / f"epochs_{NEPOCHS}"))
    pp.add_argument("--tda-root",  default=str(TDA_RESULTS_ROOT / f"epochs_{NEPOCHS}"))
    pp.add_argument("--out-dir",   default=str(POOLED_OUT_ROOT))
    pp.add_argument("--n-classes", type=int, default=N_CLASSES)

    tb = sub.add_parser("tables",
                        help="build all three LaTeX tables (TDA perf, no-TDA "
                             "perf, Base-vs-TDA overfitting) from the SAME "
                             "summary-CSV pair, so they cannot disagree")
    tb.add_argument("--tda-summary",  default=str(TDA_SUMMARY_CSV))
    tb.add_argument("--base-summary", default=str(BASE_SUMMARY_CSV))
    tb.add_argument("--out-dir",      default=str(TABLES_OUT_DIR))

    st = sub.add_parser("stats",
                        help="pooled Wilcoxon signed-rank + sign tests across "
                             "all backbones (reads engine per_fold_summary.csv)")
    cb = sub.add_parser("calibration",
                        help="pool per-fold calibration_modes.csv and compare "
                             "scalar vs per-class/affine post-hoc calibration")
    
    cb.add_argument("--cv-root", default=str(OUT_ROOT))
    cb.add_argument("--out-dir", default=str(CALIB_OUT_DIR))

    st.add_argument("--cv-root", default=str(OUT_ROOT))
    st.add_argument("--out-dir", default=str(TABLES_OUT_DIR))

    ap = sub.add_parser("all", help="run engine then pooled")
    ap.add_argument("--base-root", default=str(BASE_RESULTS_ROOT / f"epochs_{NEPOCHS}"))
    ap.add_argument("--tda-root",  default=str(TDA_RESULTS_ROOT / f"epochs_{NEPOCHS}"))
    ap.add_argument("--out-dir",   default=str(POOLED_OUT_ROOT))
    ap.add_argument("--n-classes", type=int, default=N_CLASSES)
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "engine":
        run_engine()
        run_calibration(str(OUT_ROOT), str(CALIB_OUT_DIR))

    elif args.cmd == "importance":
        plot_importance_means(verbose=True)

    elif args.cmd == "pooled":
        run_pooled(args.base_root, args.tda_root, args.out_dir, args.n_classes)

    elif args.cmd == "tables":
        run_tables(args.tda_summary, args.base_summary, args.out_dir)

    elif args.cmd == "stats":
        run_stats(args.cv_root, args.out_dir)

    elif args.cmd == "calibration":
        run_calibration(args.cv_root, args.out_dir)

    elif args.cmd == "all":
        run_engine()
        run_calibration(str(OUT_ROOT), str(CALIB_OUT_DIR))
        print("\n" + "=" * 60 + "\n[all] engine done; running pooled route\n")
        run_pooled(args.base_root, args.tda_root, args.out_dir, args.n_classes)

    elif args.cmd == "plots":
        cfg = PlotConfig(nepochs=NEPOCHS, split=SPLIT, _summary_root=OUT_ROOT)
        ReportRunner(cfg).run()

    
if __name__ == "__main__":
    main()
