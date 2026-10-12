"""
main.py
=======

Single hardwired entry point for the MID topological-deep-learning
classification pipeline. Every value the five stages need is set in the CONFIG
block below -- you never have to pass paths or hyperparameters on the command
line.

TWO WAYS TO CHOOSE WHICH STAGES RUN
-----------------------------------
1. No arguments -- run whatever is flipped True in CONFIG['RUN']:

       python main.py

   (classify defaults to False there, so a bare run never retrains by accident.)

2. Name one or more stages -- runs exactly those, ignoring the RUN flags, so
   you can run stages one at a time without editing the file:

       python main.py finalize_all
       python main.py pd_deepsets
       python main.py finalize_all pd_deepsets      # several, in the order given
       python main.py classify                      # retrain only when you ask

   Stage names: classify | finalize_all | finalize_one | pd_deepsets | fold_support
   (hyphens also accepted, e.g. finalize-all). `python main.py --list` prints them.

Either way, each stage reads its settings from its sub-dict in CONFIG.

Directory layout:

    tda_classifier/                  <- this file + the driver scripts live here
        main.py
        main_classification.py
        finalize_all.py
        run_pd_deepsets_all_backbones.py
        fold_class_support.py
        classification_python_codes/ <- the package the drivers import from
            finalize_cv_figures.py   (authoritative finalize implementation)
            making_plots.py  models.py  preparing_data_class.py
            pd_deepsets_viz.py  tsai_classifier_VAE_class.py

Which stages run is controlled by the RUN flags in CONFIG. They are ordered as
you normally run them; flip a flag to False to skip that stage. A stage's own
settings live in its sub-dict and are ignored when its RUN flag is False.
"""

from __future__ import annotations

import importlib
import os
import sys


# ===========================================================================
# CONFIG  --  edit these; there are no command-line arguments.
# ===========================================================================

# The base results folder used by the training stage and reused as the default
# root for the finalize stages (its epochs_<N> subtree).
BASE_FOLDER = "../../project_results/tsai_yes_tda_gpd_Results/classification_models/"
N_EPOCHS    = 900
EPOCHS_ROOT = os.path.join(BASE_FOLDER, f"epochs_{N_EPOCHS}")   # holds <backbone>_val_.../fold_*/


def _default_device() -> str:
    """Return 'mps', 'cuda' or 'cpu', whichever is available first."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


_DEVICE = _default_device()

CONFIG = {

    # -- which stages to run when NO stage is named on the command line ----
    #    (a CLI stage name overrides these for that run; see bottom of file).
    #    classify defaults to False so `python main.py` never retrains by
    #    accident -- turn it on explicitly when you actually want to train.
    "RUN": {
        "classify":      False,  # train + 5-fold CV all backbones, then aggregate
        "finalize_all":  True,   # figures + metrics for every backbone under EPOCHS_ROOT
        "finalize_one":  False,  # figures + metrics for a SINGLE backbone folder (set path below)
        "pd_deepsets":   True,   # PDDeepSets diagnostic figures for every trained backbone
        "fold_support":  False,  # per-fold rare-class support diagnostics (labels only)
    },

    # -- classify (main_classification.run_all) ----------------------------
    "classify": {
        "n_epochs":    N_EPOCHS,
        "seed":        42,
        "min_epoch":   1,
        "base_folder": BASE_FOLDER,
    },

    # -- finalize_all (finalize_all.main) ----------------------------------
    "finalize_all": {
        "root":      EPOCHS_ROOT,
        "n_classes": 4,
    },

    # -- finalize_one (classification_python_codes.finalize_cv_figures.finalize)
    #    Only used when RUN['finalize_one'] is True. Point at ONE backbone folder
    #    (the dir that directly contains fold_0/, fold_1/, ...).
    "finalize_one": {
        "backbone_folder": os.path.join(EPOCHS_ROOT, "RNNAttention_val_0.2_test_0.1"),
        "n_classes": 4,
    },

    # -- pd_deepsets (run_pd_deepsets_all_backbones.run_all) ----------------
    "pd_deepsets": {
        "model_root":      EPOCHS_ROOT + os.sep,
        "summary_root":    "../../project_results/tsai_comparing_models_gpd_results/nepochs_900/test_cv/summary_plots/",
        "fold":            0,
        "which":           "test",     # train | valid | test
        "batch_idx":       0,
        "sample_in_batch": 0,
        "prior_which":     None,        # None | train | valid | test
        "bs":              8,
        "device":          _DEVICE,     # cpu | mps | cuda (auto-detected)
    },

    # -- fold_support (fold_class_support.run_all) -------------------------
    #    counts=None -> load labels from dv_augm.h5; or set explicit per-class
    #    totals, e.g. [38, 80, 42, 22].
    "fold_support": {
        "counts": None,
    },
}


# ===========================================================================
# Stage runners. Imports are lazy/per-stage so a stage whose heavy dependency
# is missing (e.g. tsai/torch for classify/pd_deepsets) never blocks a stage
# that only needs pandas/sklearn (finalize / fold_support).
# ===========================================================================
def _run_classify(cfg):
    importlib.import_module("main_classification").run_all(
        n_epochs=cfg["n_epochs"],
        seed=cfg["seed"],
        min_epoch=cfg["min_epoch"],
        base_folder=cfg["base_folder"],
    )


def _run_finalize_all(cfg):
    importlib.import_module("finalize_all").main(
        cfg["root"], n_classes=cfg["n_classes"])


def _run_finalize_one(cfg):
    # package copy -- same implementation finalize_all uses
    importlib.import_module(
        "classification_python_codes.finalize_cv_figures").finalize(
        cfg["backbone_folder"], n_classes=cfg["n_classes"])


def _run_pd_deepsets(cfg):
    importlib.import_module("run_pd_deepsets_all_backbones").run_all(cfg)


def _run_fold_support(cfg):
    importlib.import_module("fold_class_support").run_all(counts=cfg["counts"])


# stage key -> (runner, config sub-dict key, human label)
_STAGES = [
    ("classify",     _run_classify,     "classify",     "Train + 5-fold CV all backbones, then aggregate"),
    ("finalize_all", _run_finalize_all, "finalize_all", "Figures + metrics for every backbone under the root"),
    ("finalize_one", _run_finalize_one, "finalize_one", "Figures + metrics for a single backbone folder"),
    ("pd_deepsets",  _run_pd_deepsets,  "pd_deepsets",  "PDDeepSets diagnostic figures for every trained backbone"),
    ("fold_support", _run_fold_support, "fold_support", "Per-fold rare-class support diagnostics"),
]


def _stage_order():
    return [k for (k, *_ ) in _STAGES]


def _normalize(name):
    """Accept hyphens or underscores: 'finalize-all' -> 'finalize_all'."""
    return name.strip().replace("-", "_")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    valid = _stage_order()

    # --list / -l / --help: show the stages and exit.
    if any(a in ("--list", "-l", "--help", "-h") for a in argv):
        print("Stages (run order):")
        for k, _fn, _ck, label in _STAGES:
            default = "on" if CONFIG["RUN"].get(k) else "off"
            print(f"  {k:<14} [{default} by default]  {label}")
        print("\nNo args -> uses CONFIG['RUN']. "
              "Name stages to run exactly those, e.g. `python main.py finalize_all`.")
        return

    # Decide the run set. CLI stage names OVERRIDE the RUN flags for this run.
    if argv:
        requested = [_normalize(a) for a in argv]
        unknown = [a for a in requested if a not in valid]
        if unknown:
            print(f"[main] Unknown stage(s): {', '.join(unknown)}")
            print(f"[main] Valid stages: {', '.join(valid)}")
            return
        selected = requested          # preserve the order the user typed
        source = "command line"
    else:
        # No args: use the RUN flags, in canonical stage order.
        selected = [k for k in valid if CONFIG["RUN"].get(k)]
        source = "CONFIG['RUN']"

    if not selected:
        print("[main] No stages selected. Either name a stage on the command "
              "line (e.g. `python main.py finalize_all`) or set a flag to True "
              "in CONFIG['RUN'].")
        return

    runner_of = {k: fn for (k, fn, _ck, _lbl) in _STAGES}
    label_of  = {k: lbl for (k, _fn, _ck, lbl) in _STAGES}

    print(f"[main] Stages to run (from {source}): {', '.join(selected)}")
    results = {"ok": [], "failed": []}
    for key in selected:
        print("\n" + "=" * 70)
        print(f"[stage: {key}] {label_of[key]}")
        print("=" * 70)
        try:
            runner_of[key](CONFIG[key])
            results["ok"].append(key)
        except Exception as e:
            # Don't let one stage's failure abort the rest of the pipeline.
            import traceback
            print(f"[stage: {key}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results["failed"].append(key)

    print("\n" + "=" * 70)
    print(f"[main] done. ok={results['ok']}  failed={results['failed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
