"""
finalize_cv_figures.py
======================

Turns the per-fold saved predictions into the THREE figure styles.
Run it once per backbone folder AFTER a CV run has produced, in each fold,
a file:  <backbone_folder>/fold_<k>/fold_predictions.npz
with keys: test_probs, test_targets (and train_/valid_ too).

    python finalize_cv_figures.py  /path/to/<backbone_folder>   [n_classes]

Output goes to <backbone_folder>/cv_final/ .

The THREE options, each in its own function so the mapping is explicit:

  OPTION 1  fig_mean_band_roc()      -> mean ROC across folds + shaded +/-1 sd band
  OPTION 2  fig_representative()     -> the single fold nearest the CV-mean AUC
  OPTION 3  fig_pooled_oof()         -> ROC + confusion + calibration on the
                                        concatenation of all folds' test preds
                                        (each country held out exactly once)

PREREQUISITE (the step that makes all three possible):
  In MakingPlots.plot_confusion_matrix_and_roc_auc_curve, after the per-split
  probabilities/targets are computed, save them:

      np.savez_compressed(
          os.path.join(self.folder, "fold_predictions.npz"),
          train_probs=..., train_targets=...,
          valid_probs=..., valid_targets=...,
          test_probs=...,  test_targets=...,
      )

  Without that file the three options cannot be built -- they are just three
  different reads of those saved arrays.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Load every fold saved predictions for a backbone.
# ---------------------------------------------------------------------------
def load_folds(backbone_folder):
    files = sorted(glob.glob(os.path.join(backbone_folder, "fold_*", "fold_predictions.npz")))
    if not files:
        raise FileNotFoundError(
            f"No fold_*/fold_predictions.npz under {backbone_folder}. "
            f"Did the SAVE_PREDS step run inside MakingPlots?")
    folds = [dict(np.load(f)) for f in files]
    print(f"loaded {len(folds)} folds from {backbone_folder}")
    return folds


def macro_auc(probs, tgts, n_classes):
    yb = label_binarize(tgts, classes=list(range(n_classes)))
    vals = []
    for i in range(n_classes):
        if yb[:, i].sum() == 0:           # class absent in this test slice
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i])
        vals.append(auc(fpr, tpr))
    return float(np.mean(vals)) if vals else float("nan")


# ===========================================================================
# OPTION 1 -- mean ROC across folds + shaded +/- 1 sd band
# ===========================================================================
def fig_mean_band_roc(bb, folds, n_classes, out):
    """Each fold's macro-ROC is interpolated onto a common FPR grid; we plot the
    mean curve and a +/-1 sd band. Legend reports mean AUC +/- sd across folds."""
    grid = np.linspace(0, 1, 200)
    per_fold_tpr, per_fold_auc = [], []
    for f in folds:
        p, t = f["test_probs"], f["test_targets"]
        yb = label_binarize(t, classes=list(range(n_classes)))
        tprs = []
        for i in range(n_classes):
            if yb[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(yb[:, i], p[:, i])
            tprs.append(np.interp(grid, fpr, tpr))
        if tprs:
            mtpr = np.mean(tprs, axis=0)
            mtpr[0] = 0.0
            per_fold_tpr.append(mtpr)
            per_fold_auc.append(auc(grid, mtpr))
    if not per_fold_tpr:
        print("  [option1] no usable folds"); return
    M = np.vstack(per_fold_tpr)
    mean, sd = M.mean(0), M.std(0)
    a_mean, a_sd = np.mean(per_fold_auc), np.std(per_fold_auc)

    plt.figure(figsize=(6, 6))
    plt.plot(grid, mean, lw=2, label=f"mean ROC (AUC = {a_mean:.3f} $\\pm$ {a_sd:.3f})")
    plt.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                     alpha=0.25, label="$\\pm$1 sd across folds")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{bb} — ROC \n mean $\\pm$ sd across folds")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  [option1] mean+band ROC -> {out}  (AUC {a_mean:.3f} +/- {a_sd:.3f})")


# ===========================================================================
# OPTION 2 -- the single representative fold (nearest the CV-mean test AUC)
# ===========================================================================
def fig_representative(bb, folds, n_classes, out_roc, out_cm):
    aucs = np.array([macro_auc(f["test_probs"], f["test_targets"], n_classes) for f in folds])
    rep = int(np.argmin(np.abs(aucs - np.nanmean(aucs))))
    p, t = folds[rep]["test_probs"], folds[rep]["test_targets"]

    # ROC of the representative fold
    yb = label_binarize(t, classes=list(range(n_classes)))
    plt.figure(figsize=(6, 6))
    for i in range(n_classes):
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], p[:, i])
        plt.plot(fpr, tpr, label=f"class {i} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{bb} — ROC — representative fold {rep} \n (test AUC = {aucs[rep]:.3f})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_roc, dpi=130); plt.close()

    # confusion of the representative fold
    cm = confusion_matrix(t, p.argmax(1), labels=list(range(n_classes)))
    _plot_cm(cm, f"{bb} — Confusion — representative fold {rep}", out_cm)
    print(f"  [option2] representative = fold {rep} (AUC {aucs[rep]:.3f}); "
          f"ROC -> {out_roc}, confusion -> {out_cm}")
    return rep


# ===========================================================================
# OPTION 3 -- pooled out-of-fold: concatenate all folds' TEST predictions
#             (each country appears exactly once, only when held out)
# ===========================================================================
def fig_pooled_oof(bb, folds, n_classes, out_roc, out_cm, out_cal):
    P = np.concatenate([f["test_probs"] for f in folds], axis=0)
    T = np.concatenate([f["test_targets"] for f in folds], axis=0)
    print(f"  [option3] pooled out-of-fold: {len(T)} held-out predictions "
          f"(every country once)")

    # ROC
    yb = label_binarize(T, classes=list(range(n_classes)))
    plt.figure(figsize=(6, 6))
    for i in range(n_classes):
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], P[:, i])
        plt.plot(fpr, tpr, label=f"class {i} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"{bb} — ROC — pooled out-of-fold \n (each country held out once)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_roc, dpi=130); plt.close()

    # confusion
    cm = confusion_matrix(T, P.argmax(1), labels=list(range(n_classes)))
    _plot_cm(cm, f"{bb} — Confusion — pooled out-of-fold", out_cm)

    # calibration (reliability) on pooled predictions
    _plot_calibration(P, T, out_cal, title=f"{bb} — Calibration — pooled out-of-fold")
    print(f"  [option3] ROC -> {out_roc}, confusion -> {out_cm}, calibration -> {out_cal}")


# ---------------------------------------------------------------------------
# small shared plotters
# ---------------------------------------------------------------------------
def _plot_cm(cm, title, out):
    plt.figure(figsize=(5, 4.2))
    plt.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, int(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.colorbar(); plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def _plot_calibration(probs, tgts, out, n_bins=10, title="Calibration"):
    conf = probs.max(1)
    correct = (probs.argmax(1) == tgts).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = (conf >= bins[b]) & (conf < bins[b + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    plt.plot(xs, ys, marker="o", label="model")
    plt.xlabel("Mean predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title(title); plt.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


# ===========================================================================
# METRICS TABLE -- quadratic-weighted kappa + per-class support + per-class AUC
#   * per-fold (mean +/- sd across folds), AND
#   * pooled out-of-fold (each country once)
# Quadratic-weighted kappa is the natural metric for an ORDINAL Likert target:
# it credits predictions that are CLOSE on the belligerency scale (3->2 counts
# as a much smaller error than 3->0), which plain accuracy/AUC ignore.
# ===========================================================================
def _qwk(y_true, y_pred, n_classes):
    """Quadratic-weighted Cohen's kappa over the fixed label set."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic",
                             labels=list(range(n_classes)))

def _per_class_auc(probs, tgts, n_classes):
    yb = label_binarize(tgts, classes=list(range(n_classes)))
    out = {}
    for i in range(n_classes):
        if yb[:, i].sum() == 0 or yb[:, i].sum() == len(tgts):
            out[i] = np.nan          # class absent (or all) in this slice
        else:
            fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i]); out[i] = auc(fpr, tpr)
    return out

def metrics_table(folds, n_classes, out_dir):
    # ---- per-fold metrics ----
    rows = []
    for k, f in enumerate(folds):
        p, t = f["test_probs"], f["test_targets"]
        pred = p.argmax(1)
        row = {"fold": k, "test_n": int(len(t)),
               "accuracy": float((pred == t).mean()),
               "qw_kappa": float(_qwk(t, pred, n_classes)),
               "macro_auc": macro_auc(p, t, n_classes)}
        pca = _per_class_auc(p, t, n_classes)
        for c in range(n_classes):
            row[f"support_c{c}"] = int((t == c).sum())
            row[f"auc_c{c}"] = pca[c]
        rows.append(row)
    per_fold = pd.DataFrame(rows)

    # ---- mean +/- sd across folds ----
    metric_cols = ["accuracy", "qw_kappa", "macro_auc"] + \
                  [f"auc_c{c}" for c in range(n_classes)]
    summary = {}
    for col in metric_cols:
        summary[col + "_mean"] = float(np.nanmean(per_fold[col]))
        summary[col + "_std"]  = float(np.nanstd(per_fold[col]))
    # support: report total across folds (each country once) + min per fold
    for c in range(n_classes):
        summary[f"support_c{c}_total"] = int(per_fold[f"support_c{c}"].sum())
        summary[f"support_c{c}_min_fold"] = int(per_fold[f"support_c{c}"].min())

    # ---- pooled out-of-fold (each country once) ----
    P = np.concatenate([f["test_probs"] for f in folds], axis=0)
    T = np.concatenate([f["test_targets"] for f in folds], axis=0)
    pred = P.argmax(1)
    pooled = {"accuracy": float((pred == T).mean()),
              "qw_kappa": float(_qwk(T, pred, n_classes)),
              "macro_auc": macro_auc(P, T, n_classes)}
    pca = _per_class_auc(P, T, n_classes)
    for c in range(n_classes):
        pooled[f"support_c{c}"] = int((T == c).sum())
        pooled[f"auc_c{c}"] = pca[c]

    # ---- write CSVs ----
    per_fold.to_csv(os.path.join(out_dir, "metrics_per_fold.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "metrics_cv_summary.csv"), index=False)
    pd.DataFrame([pooled]).to_csv(os.path.join(out_dir, "metrics_pooled_oof.csv"), index=False)

    # ---- LaTeX: headline metrics (CV mean +/- sd) + per-class support/AUC ----
    L = [r"% Auto-generated by finalize_cv_figures.py (metrics_table)",
         r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Cross-validated performance (mean $\pm$ sd across folds). "
         r"Quadratic-weighted $\kappa$ credits ordinally-close predictions, "
         r"appropriate for the four-degree belligerency scale. Per-class AUC is "
         r"shown with each class's total held-out support; the minority class "
         r"(c3) rests on few cases, so its estimate is higher-variance.}",
         r"\label{tab:cv_metrics}",
         r"\begin{tabular}{lc}", r"\toprule", r"Metric & Value \\", r"\midrule",
         f"Accuracy & {summary['accuracy_mean']:.3f} $\\pm$ {summary['accuracy_std']:.3f} \\\\",
         f"Quadratic-weighted $\\kappa$ & {summary['qw_kappa_mean']:.3f} $\\pm$ {summary['qw_kappa_std']:.3f} \\\\",
         f"Macro AUC & {summary['macro_auc_mean']:.3f} $\\pm$ {summary['macro_auc_std']:.3f} \\\\",
         r"\midrule",
         r"\multicolumn{2}{l}{\textit{Per-class AUC (support, total held-out):}} \\"]
    for c in range(n_classes):
        m = summary[f"auc_c{c}_mean"]; s = summary[f"auc_c{c}_std"]
        sup = summary[f"support_c{c}_total"]; mn = summary[f"support_c{c}_min_fold"]
        mstr = "--" if np.isnan(m) else f"{m:.3f} $\\pm$ {s:.3f}"
        L.append(f"class {c} (n={sup}, min/fold={mn}) & {mstr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(out_dir, "metrics_cv_summary.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    # ---- console ----
    print("  [metrics] CV mean +/- sd:")
    print(f"    accuracy   = {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}")
    print(f"    qw_kappa   = {summary['qw_kappa_mean']:.3f} +/- {summary['qw_kappa_std']:.3f}")
    print(f"    macro_auc  = {summary['macro_auc_mean']:.3f} +/- {summary['macro_auc_std']:.3f}")
    print(f"    pooled OOF: acc={pooled['accuracy']:.3f}, qwk={pooled['qw_kappa']:.3f}, "
          f"macroAUC={pooled['macro_auc']:.3f}")
    print(f"    per-class support (total): " +
          ", ".join(f"c{c}={summary[f'support_c{c}_total']}" for c in range(n_classes)))
    print(f"  [metrics] wrote metrics_per_fold.csv, metrics_cv_summary.csv, "
          f"metrics_pooled_oof.csv, metrics_cv_summary.tex -> {out_dir}")
    return per_fold, summary, pooled


# ---------------------------------------------------------------------------
def finalize(backbone_folder, n_classes=4):
    out = os.path.join(backbone_folder, "cv_final")
    os.makedirs(out, exist_ok=True)
    folds = load_folds(backbone_folder)

    # Backbone name only, e.g. "RNNAttention_val_0.2_test_0.1" -> "RNNAttention".
    # Split off everything from the "_val" suffix onward; tolerate a trailing
    # slash and the "-val_" variant. Falls back to the full folder name.
    base = os.path.basename(backbone_folder.rstrip("/"))
    bb = base
    for marker in ("_val_", "_val", "-val_"):
        if marker in base:
            bb = base.split(marker)[0]
            break

    print("\nOPTION 1 — mean +/- band:")
    fig_mean_band_roc(bb, folds, n_classes, os.path.join(out, "roc_mean_band.png"))

    print("OPTION 2 — representative fold:")
    fig_representative(bb, folds, n_classes,
                       os.path.join(out, "roc_representative.png"),
                       os.path.join(out, "confusion_representative.png"))

    print("OPTION 3 — pooled out-of-fold:")
    fig_pooled_oof(bb, folds, n_classes,
                   os.path.join(out, "roc_pooled_oof.png"),
                   os.path.join(out, "confusion_pooled_oof.png"),
                   os.path.join(out, "calibration_pooled_oof.png"))

    print("METRICS — quadratic-weighted kappa + per-class support/AUC:")
    metrics_table(folds, n_classes, out)

    print(f"\nAll finalized figures in: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python finalize_cv_figures.py <backbone_folder> [n_classes]")
        sys.exit(1)
    folder = sys.argv[1]
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    finalize(folder, n_classes=nc)
