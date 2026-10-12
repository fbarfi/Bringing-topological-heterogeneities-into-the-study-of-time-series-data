"""
Batch driver: PDDeepSets diagnostic figures for every backbone.

This does NOT train or evaluate anything. For each backbone it loads that
backbone's already-trained model (one fold), runs a single forward pass of the
PD encoder, and writes the three PDDeepSets figure groups (attention mass by
homology dimension, attention sharpness, and the TDA-prior / PoE figures) into
a per-backbone subdirectory placed as a sibling of the calibration plots:

    <SUMMARY_ROOT>/pd_deepsets/<backbone>/

The attention weights and prior are learned quantities, so each figure is
specific to the backbone that produced it; the backbone name is threaded into
every plot title and filename. Run this AFTER main_classification.py has
produced trained models.

Usage (from the tda_classifier/ directory):

    python run_pd_deepsets_all_backbones.py

Optional flags let you override the model root, the output root, the fold, and
the split/sample selectors.
"""
from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import torch

from classification_python_codes.pd_deepsets_viz import PDDeepSetsPlots


# ----------------------------------------------------------------------------
# Defaults. These mirror the training config in main_classification.py; if you
# change the data config there, change it here too (or the rebuilt PD tensors
# will not correspond to the loaded weights).
# ----------------------------------------------------------------------------
MODEL_ROOT = "../../project_results/tsai_yes_tda_gpd_Results/classification_models/epochs_900/"
SUMMARY_ROOT = "../../project_results/tsai_comparing_models_gpd_results/nepochs_900/test_cv/summary_plots/"

FEATURES = ['revleader', 'new_oil_state', 'coldwar', 'majorpower', 'logoil',
            'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity']
TDA_FEATURES = ["Amplitude", "Entropy", "L1", "C1", "L2", "pc1"]
PD_FOLDER = "../../project_results/gudhi_multiple_tda_results_H_4/PersistenceDiagrams/"

PKL_NAME = "trained_classification_model.pkl"


def discover_backbones(model_root: str, fold: int):
    """Yield (backbone, model_dir) for every backbone folder that has a trained
    model for the requested fold. Backbone name is parsed from the folder name
    as everything before '_val_' (e.g. 'RNNAttention_val_0.2_test_0.1')."""
    root = Path(model_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MODEL_ROOT does not exist: {root}")

    found = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or "_val_" not in d.name:
            continue
        backbone = d.name.split("_val_")[0]
        model_dir = d / f"fold_{fold}" / "models"
        pkl = model_dir / PKL_NAME
        if pkl.is_file():
            found.append((backbone, str(model_dir) + os.sep))
        else:
            print(f"[skip] {backbone}: no {PKL_NAME} at {pkl}")
    return found


def build_args(model_path: str, out_folder: str, bs: int, device: torch.device,
               backbone: str) -> dict:
    return {
        'backbone_model': backbone,  # PDDeepSetsPlots.run() reads args["backbone_model"]
        'val_size': 0.2,
        'test_size': 0.1,
        'hidden_size': 128,
        'num_layers': 2,
        'n_years': range(1945, 2011),
        'Hidx': str(1234),
        'bs': bs,
        'features': FEATURES,
        'feature_dim': len(FEATURES),
        'tda_features': TDA_FEATURES,
        'tda_features_dim': len(TDA_FEATURES),
        'pd_folder': PD_FOLDER,
        'pd_out_dim': 128,
        'vae_latent_dim': 128,
        'tda_mode': 'encode',
        'folder': out_folder,       # where THIS backbone's plots go
        'model_path': model_path,   # where THIS backbone's .pkl lives
        'device': device,
    }


def run_all(cfg=None, *, model_root=None, summary_root=None, fold=0,
            which="test", batch_idx=0, sample_in_batch=0, prior_which=None,
            bs=8, device="cpu"):
    """Generate PDDeepSets figures for every trained backbone.

    Call with either a config dict (cfg={...}, as main.py does) OR explicit
    keyword args. Values in cfg override the keyword defaults. `device` may be
    a string ('cpu'|'mps'|'cuda') or a torch.device.
    """
    if cfg:
        model_root      = cfg.get("model_root", model_root)
        summary_root    = cfg.get("summary_root", summary_root)
        fold            = cfg.get("fold", fold)
        which           = cfg.get("which", which)
        batch_idx       = cfg.get("batch_idx", batch_idx)
        sample_in_batch = cfg.get("sample_in_batch", sample_in_batch)
        prior_which     = cfg.get("prior_which", prior_which)
        bs              = cfg.get("bs", bs)
        device          = cfg.get("device", device)

    if model_root is None:
        model_root = MODEL_ROOT
    if summary_root is None:
        summary_root = SUMMARY_ROOT

    # device: accept a string or an already-built torch.device
    dev_name = device if isinstance(device, str) else str(device)
    device = torch.device(dev_name)
    if dev_name == "mps" and not torch.backends.mps.is_available():
        print("[warn] MPS not available; falling back to CPU.")
        device = torch.device("cpu")

    pd_deepsets_root = os.path.join(summary_root, "pd_deepsets")
    os.makedirs(pd_deepsets_root, exist_ok=True)

    backbones = discover_backbones(model_root, fold)
    if not backbones:
        raise RuntimeError(f"No trained models found under {model_root} for fold {fold}.")

    print(f"[info] {len(backbones)} backbone(s) to plot; output under {pd_deepsets_root}")

    ok, failed = [], []
    for backbone, model_path in backbones:
        out_folder = os.path.join(pd_deepsets_root, backbone) + os.sep
        os.makedirs(out_folder, exist_ok=True)
        print(f"\n===== {backbone} =====")
        print(f"  model: {model_path}")
        print(f"  out:   {out_folder}")
        try:
            args = build_args(model_path, out_folder, bs=bs, device=device,
                              backbone=backbone)
            viz = PDDeepSetsPlots(args)
            viz.load_model()
            paths = viz.run(
                which=which,
                batch_idx=batch_idx,
                sample_in_batch=sample_in_batch,
                prior_which=prior_which,
                years=list(args['n_years']),
            )
            for name, pth in paths.items():
                print(f"    {name}: {pth}")
            ok.append(backbone)
        except Exception as e:
            print(f"  [error] {backbone}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(backbone)

    print("\n================ SUMMARY ================")
    print(f"  succeeded ({len(ok)}): {', '.join(ok) if ok else '-'}")
    print(f"  failed    ({len(failed)}): {', '.join(failed) if failed else '-'}")
    return {"ok": ok, "failed": failed}


def main(argv=None):
    p = argparse.ArgumentParser(description="PDDeepSets figures for all backbones.")
    p.add_argument("--model-root", default=MODEL_ROOT,
                   help="Root holding <backbone>_val_.../fold_k/models/ trees.")
    p.add_argument("--summary-root", default=SUMMARY_ROOT,
                   help="summary_plots root; output goes to <summary-root>/pd_deepsets/<backbone>/.")
    p.add_argument("--fold", type=int, default=0,
                   help="Which fold's trained model represents each backbone (default 0).")
    p.add_argument("--which", default="test", choices=["train", "valid", "test"])
    p.add_argument("--batch-idx", type=int, default=0)
    p.add_argument("--sample-in-batch", type=int, default=0)
    p.add_argument("--prior-which", default=None, choices=[None, "train", "valid", "test"])
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    a = p.parse_args(argv)
    run_all(model_root=a.model_root, summary_root=a.summary_root, fold=a.fold,
            which=a.which, batch_idx=a.batch_idx, sample_in_batch=a.sample_in_batch,
            prior_which=a.prior_which, bs=a.bs, device=a.device)


if __name__ == "__main__":
    main()
