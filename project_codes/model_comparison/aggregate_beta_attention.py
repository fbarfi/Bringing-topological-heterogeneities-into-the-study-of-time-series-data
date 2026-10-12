"""
Fold-aggregated conditioning-attention (beta) diagnostics -- standalone.

Produces, per backbone, the fold-mean +/- std of:
  - beta sharpness over time:  effective #points and entropy per year
  - beta mass by homology dim over time:  attention mass on each H_k per year

beta is the point-level conditioning attention (model._last_point_attn), the
discriminative attention introduced by the point cross-attention conditioner.

Aggregation convention (mirrors the accuracy mean+/-std reporting): summarize
per-year WITHIN each fold first (effn, entropy, mass_d), THEN take mean/std
across folds. Raw beta is never averaged across folds -- that would blur
distinct fitted distributions and understate concentration.

Run directly:
    python aggregate_beta_attention.py --which test --device cpu
    python aggregate_beta_attention.py --models TSiT --device mps
Run from the SAME working directory training used: PreparingDataClass reads
data via relative paths.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tsai.all import load_learner


# ============================== CONFIG ==============================
HERE = Path(__file__).resolve().parent

NEPOCHS  = 900
N_SPLITS = 5
SEED     = 42
VAL_FRAC = 0.15
N_DIMS   = 4                 # homology dims 1..4 carried by the conditioner
N_YEARS  = (1945, 2011)

VAL_SIZE  = 0.2              # naming tag only (matches trained-model dir layout)
TEST_SIZE = 0.1             # naming tag only
BS        = 64              # eval batch size; affects speed only, not results

TSAI_MODELS = [
    "RNNAttention", "LSTMAttention", "GRUAttention", "RNNPlus",
    "TSSequencerPlus", "TSiT", "TSiTPlus", "ResNetPlus",
    "InceptionTimePlus", "XCMPlus",
]

# Note: feature lists and the dv/idv HDF paths are hardcoded inside
# PreparingDataClass (which reads them via its own relative paths), so they are
# not configured here -- run from the same working directory training used.
TDA_RESULTS_ROOT = (HERE / "../../project_results/tsai_yes_tda_gpd_Results/classification_models").resolve()
TDA_CODE_ROOT    = (HERE / "../../project_codes/tda_classifier").resolve()
PD_FOLDER        = (HERE / "../../project_results/gudhi_multiple_tda_results_H_4/PersistenceDiagrams/").resolve()
HIDX             = "1234"
OUT_ROOT         = (HERE / f"../../project_results/tsai_comparing_models_gpd_results/nepochs_{NEPOCHS}/beta_attention").resolve()

# PreparingDataClass lives in the TDA training package; make it importable both
# for this import and for unpickling the custom model class stored in each pkl.
if str(TDA_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(TDA_CODE_ROOT))
from comparing_codes.tda_preparing_data_class import PreparingDataClass


# ============================ DATA / MODEL ============================
def build_prep(device):
    """One PreparingDataClass instance, arrays loaded and cached once. Its cached
    `y` (from _load_align, which drops countries with incomplete panels) is the
    ONLY correct basis for folds: make_dls_fold indexes X[tr] over that same
    filtered ordering, so folds must be built from this y, not a separately
    re-derived label array."""
    prep = PreparingDataClass({
        "n_years":       list(range(*N_YEARS)),
        "val_size":      VAL_FRAC,
        "test_size":     TEST_SIZE,
        "pd_folder":     str(PD_FOLDER),
        "Hidx":          HIDX,
        "dims":          tuple(range(1, N_DIMS + 1)),
        "bs":            BS,
        "device":        device,
        "shuffle_train": False,         # eval: never shuffle
    })
    prep.load_arrays_once()
    return prep


def make_folds(y: np.ndarray):
    """Rebuild the SAME (train, valid, test) country folds used in training by
    delegating to the training pipeline's own generator -- one source of truth,
    so the folds cannot drift from what produced the pkls."""
    return PreparingDataClass.make_cv_folds(
        y, n_splits=N_SPLITS, val_frac=VAL_FRAC, seed=SEED)


def pick_loader(prep, tr, va, te, which):
    """Loaders for one fold via make_dls_fold, which pins the split to the exact
    fold indices the model was trained on. (make_dls() ignores the fold and
    always re-splits, so it must NOT be used here.) Batches are shaped
    (X, PD_pts, PD_mask, PD_dim, y) via TSDataLoaderPDHMulti.collate_batch.
    Layout: dls[0]=train, dls[1]=valid, test attached as dls.test (not dls[2])."""
    _, dls = prep.make_dls_fold(np.asarray(tr), np.asarray(va), np.asarray(te))
    return {"train": dls[0], "valid": dls[1], "test": dls.test}[which]


def pkl_path(model: str, fold: int) -> Path:
    return (TDA_RESULTS_ROOT / f"epochs_{NEPOCHS}"
            / f"{model}_val_{VAL_SIZE}_test_{TEST_SIZE}"
            / f"fold_{fold}" / "models" / "trained_classification_model.pkl").resolve()


def load_model(pkl: Path, device: torch.device):
    """Robust learner load across fastai/torch versions.

    fastai's load_learner can mask the real error as an UnboundLocalError on
    'res' when torch.load itself raises (e.g. torch>=2.6 defaulting
    weights_only=True, which cannot unpickle a full Learner). Fall back to a
    direct torch.load(weights_only=False) in that case.
    """
    try:
        return load_learner(pkl, cpu=(device.type == "cpu")).model.to(device)
    except Exception as e:
        print(f"[load] load_learner failed ({type(e).__name__}: {e}); "
              f"retrying torch.load(weights_only=False)")
        obj = torch.load(pkl, map_location="cpu", weights_only=False)
        return getattr(obj, "model", obj).to(device)


# ============================ BETA EXTRACTION ============================
def _read_beta_mask(model):
    """Fetch (beta, mask) atomically from the same source so they can't desync.
    Prefer the top-level model attrs; fall back to the conditioner submodule."""
    beta = getattr(model, "_last_point_attn", None)
    mask = getattr(model, "_last_point_mask", None)
    if beta is None or mask is None:
        cond = getattr(model, "tda_conditioner", None)
        if cond is None:
            raise AttributeError(
                "no _last_point_attn on model and no tda_conditioner submodule; "
                "this pkl is likely not a point-conditioner model -- beta undefined")
        beta = cond._last_point_attn
        mask = cond._last_point_mask
    if beta is None or mask is None:
        raise AttributeError("conditioner did not populate _last_point_attn/_mask "
                             "after forward; check the forward path sets them")
    return beta, mask


@torch.no_grad()
def fold_beta_summaries(model, loader, device, T, n_dims=N_DIMS, probe=False):
    """For one trained fold, compute per-year effn, entropy, and per-dimension
    attention mass, averaged over all samples in the given split.

    Returns arrays indexed by year (length T): effn[T], entropy[T], n[T],
    mass[n_dims, T], valid[T]. Years with no valid samples are NaN (excluded by
    the across-fold nanmean), not 0.
    """
    model.eval()
    sum_effn = np.zeros(T); sum_ent = np.zeros(T); cnt = np.zeros(T)
    sum_n = np.zeros(T)                      # accumulate, then average
    sum_mass = np.zeros((n_dims, T)); mass_cnt = np.zeros(T)
    dim_pts = np.zeros((n_dims, T))          # max #points per dim per year (for degeneracy mask)
    probed = False

    for x, pd_pts, pd_mask, pd_dim, _y in loader:
        x = x.to(device, torch.float32)
        pd_pts = pd_pts.to(device, torch.float32)
        pd_mask = pd_mask.to(device, torch.float32)
        pd_dim = pd_dim.to(device, torch.long)

        model(x, pd_pts, pd_mask, pd_dim)
        beta, mask = _read_beta_mask(model)   # beta:[B,Tm,P]  mask:[B,Tm,P]

        B, Tm, _P = beta.shape
        if Tm != T:
            raise ValueError(
                f"model time dim {Tm} != expected year count {T}. The plot x-axis "
                f"assumes position t maps to year {N_YEARS[0]}+t; a mismatch means "
                f"years are misaligned. Check N_YEARS vs the loader's sequence length.")

        if probe and not probed:
            # Whole-tensor fingerprint over ALL masked (b,t,p), not one slice.
            # Two backbones that print the same fp/eff-quantiles are producing
            # identical beta; differing values prove beta is backbone-specific.
            m = mask > 0
            bm = beta[m]
            # per-(b,t) effective-n over masked points, vectorized
            w_full = beta.masked_fill(~m, 0.0)
            denom = w_full.sum(-1, keepdim=True).clamp_min(1e-12)
            wn = w_full / denom
            effn_bt = 1.0 / wn.pow(2).sum(-1).clamp_min(1e-12)   # [B,Tm]
            n_bt = m.sum(-1).clamp_min(1)                          # [B,Tm]
            ratio = (effn_bt / n_bt)[m.any(-1)]                   # valid years only
            fp = float(bm.double().sum().item())                 # cheap content hash
            q = torch.quantile(ratio.flatten(),
                               torch.tensor([0.05, 0.5, 0.95], device=ratio.device))
            print(f"    [probe] beta fp(sum)={fp:.6f}  eff/n quantiles "
                  f"[p05,p50,p95]={[round(float(v),3) for v in q]}  "
                  f"masked_min/max beta={float(bm.min()):.4e}/{float(bm.max()):.4e}")
            probed = True

        for b in range(B):
            for t in range(T):
                mt = mask[b, t] > 0
                n = int(mt.sum())
                if n < 2:
                    continue
                w = beta[b, t][mt]
                w = w / w.sum().clamp_min(1e-12)

                sum_effn[t] += float(1.0 / w.pow(2).sum().clamp_min(1e-12))
                sum_ent[t] += float(-(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum())
                sum_n[t] += n
                cnt[t] += 1

                d_t = pd_dim[b, t][mt]                 # dims 1..n_dims; 0 = padding
                mass_here = 0.0
                for d in range(1, n_dims + 1):
                    in_d = (d_t == d)
                    md = float(w[in_d].sum())
                    sum_mass[d - 1, t] += md
                    mass_here += md
                    # points present in dim d this (b,t); PDs are shared within a
                    # year so this is ~constant across samples -- take the max.
                    dim_pts[d - 1, t] = max(dim_pts[d - 1, t], int(in_d.sum()))
                # closure check: masked weights sum to 1, so per-dim masses must
                # too (up to float error). A shortfall means pd_dim carries values
                # outside 1..n_dims (e.g. a dim-5 leak vs n_dims=4) -> silent loss.
                if abs(mass_here - 1.0) > 1e-3:
                    raise ValueError(
                        f"mass over dims 1..{n_dims} = {mass_here:.4f} != 1 at "
                        f"year-pos {t}: pd_dim has out-of-range dims; increase "
                        f"n_dims or check the PD files.")
                mass_cnt[t] += 1

    valid = cnt > 0
    def _avg(num, den):
        out = np.full(T, np.nan)
        out[valid] = num[valid] / den[valid]
        return out

    mass = np.full((n_dims, T), np.nan)
    mass[:, valid] = sum_mass[:, valid] / mass_cnt[valid]

    # Non-degenerate year: at least 2 homology dims each populated with >=2
    # points. Below that, "distribution over dims" is an accounting identity of
    # sparse diagrams (all mass forced onto the one non-empty dim), not a
    # property of attention -- so it must be excluded from the by-dim mass plot.
    nondegen = (dim_pts >= 2).sum(0) >= 2      # [T] bool

    return {
        "effn":     _avg(sum_effn, cnt),
        "entropy":  _avg(sum_ent, cnt),
        "n":        _avg(sum_n, cnt),
        "mass":     mass,
        "dim_pts":  dim_pts,                    # [n_dims, T]
        "nondegen": nondegen,                   # [T] bool
        "valid":    valid,
    }


# ============================ PLOTTING ============================
def plot_sharpness(years, effn_stack, ent_stack, out_path, backbone):
    """effn_stack, ent_stack: [n_folds, T]."""
    ent_m, ent_s = np.nanmean(ent_stack, 0), np.nanstd(ent_stack, 0)
    effn_m, effn_s = np.nanmean(effn_stack, 0), np.nanstd(effn_stack, 0)

    fig, axL = plt.subplots(figsize=(10, 4))
    cE = "tab:blue"
    axL.plot(years, ent_m, color=cE, lw=1.5, marker="o", ms=3,
             label="entropy (lower=sharper)")
    axL.fill_between(years, ent_m - ent_s, ent_m + ent_s, color=cE, alpha=0.2)
    axL.set_xlabel("year"); axL.set_ylabel("entropy (nats)", color=cE)
    axL.tick_params(axis="y", labelcolor=cE)

    axR = axL.twinx()
    cN = "tab:orange"
    axR.plot(years, effn_m, color=cN, lw=1.5, marker="o", ms=3,
             label="effective #points")
    axR.fill_between(years, effn_m - effn_s, effn_m + effn_s, color=cN, alpha=0.2)
    axR.set_ylabel("effective #points", color=cN)
    axR.tick_params(axis="y", labelcolor=cN)

    axL.set_title(f"Conditioning attention sharpness -- fold mean $\\pm$ sd ({backbone})")
    lines = axL.get_lines() + axR.get_lines()
    axL.legend(lines, [l.get_label() for l in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mass(years, mass_stack, out_path, backbone, n_dims=N_DIMS):
    """mass_stack: [n_folds, n_dims, T]."""
    fig, ax = plt.subplots(figsize=(10, 4))
    top = 0.0
    for d in range(n_dims):
        m = np.nanmean(mass_stack[:, d, :], 0)
        s = np.nanstd(mass_stack[:, d, :], 0)
        line, = ax.plot(years, m, lw=1.5, marker="o", ms=3, label=f"dim={d+1}")
        ax.fill_between(years, m - s, m + s, color=line.get_color(), alpha=0.18)
        top = max(top, np.nanmax(m + s))
    ax.set_xlabel("year"); ax.set_ylabel("conditioning attention mass")
    # Adaptive top so small but real between-backbone differences stay legible;
    # keep 0 as the floor since mass is non-negative and read relative to 0.
    ax.set_ylim(0.0, min(1.0, top * 1.10) if np.isfinite(top) and top > 0 else 1.0)
    ax.set_title(f"Conditioning attention mass by homology dim -- fold mean $\\pm$ sd ({backbone})")
    ax.legend(ncol=min(4, n_dims))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================ DRIVER ============================
def run(models, which, device, out_dir, n_dims=N_DIMS, probe=False):
    years = np.arange(*N_YEARS)
    T = len(years)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    prep = build_prep(device)
    y = prep.load_arrays_once()["y"]
    folds = make_folds(y)

    for model_name in models:
        effn_folds, ent_folds, mass_folds, nondegen_folds = [], [], [], []
        for k, (tr, va, te) in enumerate(folds):
            pkl = pkl_path(model_name, k)
            if not pkl.is_file():
                print(f"[skip] {model_name} fold {k}: no pkl")
                continue

            model = load_model(pkl, device)
            loader = pick_loader(prep, tr, va, te, which)
            s = fold_beta_summaries(model, loader, device, T,
                                    n_dims=n_dims, probe=probe)

            effn_folds.append(s["effn"])
            ent_folds.append(s["entropy"])
            mass_folds.append(s["mass"])
            nondegen_folds.append(s["nondegen"])
            with np.errstate(invalid="ignore"):
                ratio = np.nanmean(s["effn"] / s["n"])
            print(f"[ok] {model_name} fold {k}: mean effn/n = {ratio:.3f}")

        if not effn_folds:
            print(f"[warn] {model_name}: no folds; skipping plots")
            continue

        effn_stack = np.vstack(effn_folds)      # [F,T]
        ent_stack = np.vstack(ent_folds)        # [F,T]
        mass_stack = np.stack(mass_folds, 0)    # [F,n_dims,T]
        # A year is usable for by-dim mass if non-degenerate in every fold; the
        # PDs are shared across folds so this is really a per-year property, but
        # AND-ing across folds is the conservative choice.
        nondegen = np.all(np.vstack(nondegen_folds), axis=0)   # [T] bool

        bdir = out_dir / model_name
        bdir.mkdir(parents=True, exist_ok=True)
        plot_sharpness(years, effn_stack, ent_stack,
                       bdir / "condattn_sharpness_foldmean.png", model_name)
        plot_mass(years, mass_stack,
                  bdir / "condattn_mass_by_dim_foldmean.png", model_name, n_dims=n_dims)
        np.savez(bdir / "condattn_foldagg.npz",
                 years=years, effn=effn_stack, entropy=ent_stack,
                 mass=mass_stack, nondegen=nondegen)
        print(f"[done] {model_name}: wrote fold-mean sharpness + mass to {bdir}")


def _load_backbone_foldmeans(out_dir, models):
    """Read each backbone's saved npz and reduce to its across-fold mean curves.
    Returns years, and dicts model -> effn[T], entropy[T], mass[n_dims,T],
    nondegen[T]. Skips backbones whose npz is missing. Older npz without the
    'nondegen' key get an all-True mask (no degeneracy filtering)."""
    years = None
    effn, ent, mass, nondegen = {}, {}, {}, {}
    for m in models:
        p = Path(out_dir) / m / "condattn_foldagg.npz"
        if not p.is_file():
            print(f"[overlay] skip {m}: no {p.name}")
            continue
        z = np.load(p)
        if years is None:
            years = z["years"]
        effn[m] = np.nanmean(z["effn"], 0)          # [T]
        ent[m] = np.nanmean(z["entropy"], 0)        # [T]
        mass[m] = np.nanmean(z["mass"], 0)          # [n_dims, T]
        nondegen[m] = (z["nondegen"] if "nondegen" in z.files
                       else np.ones(len(z["years"]), bool))
    return years, effn, ent, mass, nondegen


def _pooled(curves):
    """Stack per-backbone curves and return (mean, sd, lo, hi) across backbones,
    NaN-aware. curves: dict model -> array[...]; stacks on a new axis 0. Years
    that are all-NaN across backbones stay NaN (excluded from plotting)."""
    A = np.stack(list(curves.values()), 0)          # [M, ...]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slice is expected
        mean = np.nanmean(A, 0)
        sd = np.nanstd(A, 0)
        lo = np.nanmin(A, 0)
        hi = np.nanmax(A, 0)
    return mean, sd, lo, hi


def plot_overlay_sharpness(out_dir, models):
    """One figure: entropy + effn pooled across backbones, band = ±1 SD across
    backbones, whiskers = min-max (mirrors the reliability-spread template).
    Each source of the pool is a backbone's across-fold mean curve, so the band
    is between-backbone disagreement, not fold noise."""
    years, effn, ent, _mass, _nd = _load_backbone_foldmeans(out_dir, models)
    if not ent:
        print("[overlay] no backbones with saved npz; nothing to plot")
        return
    M = len(ent)

    e_m, e_sd, e_lo, e_hi = _pooled(ent)
    n_m, n_sd, n_lo, n_hi = _pooled(effn)

    fig, axL = plt.subplots(figsize=(11, 5))
    cE, cN = "tab:blue", "tab:orange"

    axL.plot(years, e_m, color=cE, lw=2, marker="o", ms=3,
             label=f"entropy (pooled over {M} backbones)")
    axL.fill_between(years, e_m - e_sd, e_m + e_sd, color=cE, alpha=0.20)
    axL.vlines(years, e_lo, e_hi, color=cE, alpha=0.25, lw=0.8)
    axL.set_xlabel("year"); axL.set_ylabel("entropy (nats)", color=cE)
    axL.tick_params(axis="y", labelcolor=cE)

    axR = axL.twinx()
    axR.plot(years, n_m, color=cN, lw=2, marker="o", ms=3,
             label="effective #points (pooled)")
    axR.fill_between(years, n_m - n_sd, n_m + n_sd, color=cN, alpha=0.20)
    axR.vlines(years, n_lo, n_hi, color=cN, alpha=0.25, lw=0.8)
    axR.set_ylabel("effective #points", color=cN)
    axR.tick_params(axis="y", labelcolor=cN)

    axL.set_title(f"Conditioning attention sharpness across backbones "
                  f"-- pooled mean, band $\\pm$1 SD, whisker min-max ({M} backbones)")
    lines = axL.get_lines() + axR.get_lines()
    axL.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    fig.tight_layout()
    out = Path(out_dir) / "condattn_sharpness_all_backbones_spread.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] wrote {out}")


def plot_overlay_mass(out_dir, models, n_dims=N_DIMS):
    """One figure: per-homology-dim mass pooled across backbones, band = ±1 SD
    across backbones. One colored curve+band per dim."""
    years, _effn, _ent, mass, _nd = _load_backbone_foldmeans(out_dir, models)
    if not mass:
        print("[overlay] no backbones with saved npz; nothing to plot")
        return
    M = len(mass)

    fig, ax = plt.subplots(figsize=(11, 5))
    top = 0.0
    for d in range(n_dims):
        curves = {m: mass[m][d] for m in mass}      # model -> [T] for this dim
        m_mean, m_sd, _lo, _hi = _pooled(curves)
        line, = ax.plot(years, m_mean, lw=2, marker="o", ms=3, label=f"dim={d+1}")
        ax.fill_between(years, m_mean - m_sd, m_mean + m_sd,
                        color=line.get_color(), alpha=0.18)
        top = max(top, np.nanmax(m_mean + m_sd))

    ax.set_xlabel("year"); ax.set_ylabel("conditioning attention mass")
    ax.set_ylim(0.0, min(1.0, top * 1.10) if np.isfinite(top) and top > 0 else 1.0)
    ax.set_title(f"Conditioning attention mass by homology dim across backbones "
                 f"-- pooled mean, band $\\pm$1 SD ({M} backbones)")
    ax.legend(ncol=min(4, n_dims))
    fig.tight_layout()
    out = Path(out_dir) / "condattn_mass_all_backbones_spread.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] wrote {out}")


def plot_overlay_mass_masked(out_dir, models, n_dims=N_DIMS):
    """Same as plot_overlay_mass but degenerate years are blanked (set NaN).

    A year is degenerate when fewer than two homology dims are populated with >=2
    points: there the by-dim mass split is an accounting identity of the sparse
    persistence diagram (all mass forced onto the single non-empty dim), not a
    property of attention. Those years are dropped so the plot shows only where
    the dim-distribution is actually meaningful. Pool the per-backbone nondegen
    masks by AND (a year is shown only if every backbone finds it non-degenerate;
    since PDs are shared this is really a per-year property)."""
    years, _effn, _ent, mass, nondegen = _load_backbone_foldmeans(out_dir, models)
    if not mass:
        print("[overlay] no backbones with saved npz; nothing to plot")
        return
    M = len(mass)

    keep = np.all(np.stack(list(nondegen.values()), 0), axis=0)   # [T] bool
    n_drop = int((~keep).sum())

    fig, ax = plt.subplots(figsize=(11, 5))
    top = 0.0
    for d in range(n_dims):
        curves = {m: np.where(keep, mass[m][d], np.nan) for m in mass}
        m_mean, m_sd, _lo, _hi = _pooled(curves)
        line, = ax.plot(years, m_mean, lw=2, marker="o", ms=3, label=f"dim={d+1}")
        ax.fill_between(years, m_mean - m_sd, m_mean + m_sd,
                        color=line.get_color(), alpha=0.18)
        if np.isfinite(m_mean + m_sd).any():
            top = max(top, np.nanmax(m_mean + m_sd))

    ax.set_xlabel("year"); ax.set_ylabel("conditioning attention mass")
    ax.set_ylim(0.0, min(1.0, top * 1.10) if np.isfinite(top) and top > 0 else 1.0)
    ax.set_title(
        f"Conditioning attention mass by homology dim across backbones\n"
        f"non-degenerate years only, pooled mean $\\pm$1 SD "
        f"({M} backbones, {n_drop} degenerate yrs dropped)")
    
    ax.legend(ncol=min(4, n_dims))
    fig.tight_layout()
    out = Path(out_dir) / "condattn_mass_all_backbones_spread_nondegenerate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlay] wrote {out} ({n_drop} degenerate years masked)")


def main():
    ap = argparse.ArgumentParser(
        description="Autonomous fold-aggregated conditioning-attention (beta) diagnostics.")
    ap.add_argument("--which", default="test", choices=["train", "valid", "test"],
                    help="which split to summarize beta over")
    ap.add_argument("--device", default="cpu",
                    help="torch device string, e.g. cpu / cuda:0 / mps")
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of backbones (default: all)")
    ap.add_argument("--n-dims", type=int, default=N_DIMS)
    ap.add_argument("--probe", action="store_true",
                    help="print beta signal stats (eff/n≈1 => uniform, i.e. no "
                         "backbone-specific attention -- explains identical plots)")
    ap.add_argument("--overlay-only", action="store_true",
                    help="skip per-backbone extraction; build the cross-backbone "
                         "spread plots from already-saved condattn_foldagg.npz")
    args = ap.parse_args()

    models = args.models or TSAI_MODELS

    if not args.overlay_only:
        run(
            models=models,
            which=args.which,
            device=torch.device(args.device),
            out_dir=args.out_dir,
            n_dims=args.n_dims,
            probe=args.probe,
        )

    # cross-backbone spread figures (one plot bringing all backbones together)
    plot_overlay_sharpness(args.out_dir, models)
    plot_overlay_mass(args.out_dir, models, n_dims=args.n_dims)
    plot_overlay_mass_masked(args.out_dir, models, n_dims=args.n_dims)


if __name__ == "__main__":
    main()
