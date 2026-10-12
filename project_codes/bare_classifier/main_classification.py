
from __future__ import annotations
from typing import List, Dict, Any, Tuple

import os, sys, warnings, traceback, time, gc
from datetime import datetime
from pathlib import Path
import torch
import tsai
import pandas as pd
import numpy as np
from tsai.all import *

from classification_python_codes.making_plots import MakingPlots
from classification_python_codes.models import ModelsClass
from classification_python_codes.preparing_data_class import PreparingDataClass

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


def default_device() -> torch.device:
    """Return the first available accelerator: MPS (Apple silicon), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = default_device()

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except Exception: return pd.DataFrame()

def _last_non_nan(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty: return np.nan
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def _min_non_nan(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty: return np.nan
    s = df[col].dropna()
    return float(s.min()) if len(s) else np.nan

def extract_run_metrics(run_dir: Path, k_last: int) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    history = _read_csv_safe(run_dir / "history.csv")
    overall = _read_csv_safe(run_dir / "overall_metrics.csv")
    cfg     = _read_csv_safe(run_dir / "model_config.csv")

    backbone = None
    seed     = None
    n_epochs = None
    try:
        if not cfg.empty:
            backbone = cfg.get("backbone_model", pd.Series([None])).iloc[0]
            seed     = cfg.get("seed", pd.Series([None])).iloc[0]
            n_epochs = cfg.get("n_epochs", pd.Series([None])).iloc[0]
    except Exception:
        pass

    col_val_loss = next((c for c in history.columns if c.lower() in {"valid_loss","val_loss"}), None)
    col_tr_loss  = next((c for c in history.columns if c.lower() in {"train_loss","loss"}), None)
    col_val_auc  = next((c for c in history.columns if "valid" in c.lower() and "auc" in c.lower()), None)
    col_tr_auc   = next((c for c in history.columns if "train" in c.lower() and "auc" in c.lower()), None)
    col_val_f1   = next((c for c in history.columns if "valid" in c.lower() and "f1" in c.lower()), None)

    best_val_loss   = _min_non_nan(history, col_val_loss) if col_val_loss else np.nan
    last_train_loss = _last_non_nan(history, col_tr_loss) if col_tr_loss else np.nan
    last_valid_auc  = _last_non_nan(history, col_val_auc) if col_val_auc else np.nan
    last_train_auc  = _last_non_nan(history, col_tr_auc) if col_tr_auc else np.nan
    last_valid_f1   = _last_non_nan(history, col_val_f1) if col_val_f1 else np.nan

    test_acc = np.nan
    if not overall.empty and {"Metric","Value"}.issubset(overall.columns):
        try:
            test_acc = float(overall.loc[overall["Metric"].str.lower()=="accuracy","Value"].iloc[0])
        except Exception:
            pass

    return {
        "run_dir": str(run_dir),
        "backbone": backbone if pd.notna(backbone) else "UNKNOWN",
        "seed": int(seed) if pd.notna(seed) else np.nan,
        "n_epochs": int(n_epochs) if pd.notna(n_epochs) else np.nan,
        "best_valid_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "last_valid_auc": last_valid_auc,
        "last_train_auc": last_train_auc,
        "last_valid_f1": last_valid_f1,
        "test_accuracy": test_acc,
    }

def mean_std_table(df: pd.DataFrame, group_col: str, metrics: List[str]) -> pd.DataFrame:
    agg = df.groupby(group_col)[metrics].agg(["mean","std","count"])
    agg.columns = [f"{m}_{s}" for m,s in agg.columns]
    return agg.reset_index()

def to_latex_pm(df: pd.DataFrame,
                group_col: str,
                metrics: List[str],
                caption: str,
                label: str) -> str:
    cols = [group_col] + metrics
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[group_col])]
        for m in metrics:
            mean = row.get(f"{m}_mean", np.nan)
            std  = row.get(f"{m}_std", np.nan)
            cells.append("--" if np.isnan(mean) else f"{mean:.4f} ± {std:.4f}")
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join(cols) + " \\\\"
    hline = "\\hline"
    body  = "\n".join(rows)
    return (
        "\\begin{{table}}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{{tabular}}{{{cols_spec}}}\n"
        "{hline}\n"
        "{header}\n"
        "{hline}\n"
        "{body}\n"
        "{hline}\n"
        "\\end{{tabular}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{{table}}\n"
    ).format(
        cols_spec='l' + 'c'*len(metrics),
        hline=hline,
        header=header,
        body=body
    )

def _discover_runs_with_meta(base_folder: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Find run directories and pull metadata (backbone, seed) from model_config.csv.
    A 'run' is any dir containing history.csv.
    Returns: [(run_dir, meta_dict), ...]
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for hist in Path(base_folder).rglob("history.csv"):
        run_dir = hist.parent
        cfg = _read_csv_safe(run_dir / "model_config.csv")
        meta = {"backbone": "UNKNOWN", "seed": np.nan}
        try:
            if not cfg.empty:
                meta["backbone"] = str(cfg.get("backbone_model", pd.Series(["UNKNOWN"])).iloc[0])
                seed = cfg.get("seed", pd.Series([np.nan])).iloc[0]
                meta["seed"] = int(seed) if pd.notna(seed) else np.nan
        except Exception:
            pass
        out.append((str(run_dir), meta))
    return out


def run_cross_validation(args, n_splits=5, seed=42):
    """K-fold CV for the model, one backbone, paired folds (seed=42).
    Builds each fold's dls ONCE and reuses the SAME prebuilt tuple for both
    ModelsClass and MakingPlots, so plots/metrics use the fold's held-out test
    set."""

    prep = PreparingDataClass(args)
    cache = prep.load_arrays_once()
    folds = prep.make_cv_folds(cache["y"], n_splits=n_splits, val_frac=0.15, seed=seed)

    rows = []
    for k, (tr, va, te) in enumerate(folds):
        print(f"\n===== [ {args['backbone_model']}] fold {k+1}/{n_splits} "
              f"(train={len(tr)}, val={len(va)}, test={len(te)}) =====")

        fold_folder = os.path.join(args['folder'], f"fold_{k}/")
        fold_model_path = os.path.join(fold_folder, "models/")
        os.makedirs(fold_model_path, exist_ok=True)

        fold_args = dict(args)
        fold_args['folder'] = fold_folder
        fold_args['model_path'] = fold_model_path
        fold_args['seed'] = seed + k

        cfg = {k2: v for k2, v in fold_args.items() if k2 != '_prebuilt'}
        cfg['device'] = str(cfg.get('device', ''))
        pd.DataFrame([cfg]).to_csv(os.path.join(fold_folder, 'model_config.csv'), index=False)

        # build the fold's loaders ONCE; reuse for BOTH model and plots
        splits_dls = prep.make_dls_fold(tr, va, te)        # (splits, dls)
        fold_args['_prebuilt'] = (prep,) + splits_dls

        ts_class = ModelsClass(fold_args)
        ts_class.run_model()
        ts_class.run_test()

        # MakingPlots reuses the SAME fold_args (so _prebuilt is present) ->
        # evaluates the fold's trained model on the fold's held-out test set
        mp = MakingPlots(fold_args)
        accs, aucs = mp.plotting_fn()
        tr_acc, va_acc, te_acc = accs
        tr_auc, va_auc, te_auc = aucs
        rows.append({
            'backbone': args['backbone_model'], 'fold': k,
            'train_accuracy': tr_acc, 'valid_accuracy': va_acc, 'test_accuracy': te_acc,
            'train_auc': tr_auc, 'valid_auc': va_auc, 'test_auc': te_auc,
        })

        # release AFTER plots are done (do not clear _prebuilt before MakingPlots)
        fold_args['_prebuilt'] = None
        # _release_fold(ts_class=ts_class, mp=mp)
        ts_class = None
        mp = None

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args['folder'], 'cv_folds.csv'), index=False)
    print(f"\n===== CV SUMMARY [{args['backbone_model']}] (mean +/- std over {n_splits} folds) =====")
    for col in ['train_accuracy','valid_accuracy','test_accuracy','train_auc','valid_auc','test_auc']:
        if col in res and res[col].notna().any():
            print(f"  {col}: {res[col].mean():.4f} +/- {res[col].std():.4f}")
    return res

def aggregate_all_runs_for_publication(base_folder: str) -> str:
    """
    Auto-discovers runs, groups by backbone, computes mean±std across seeds,
    and writes raw, group summaries, and LaTeX with an auto caption.
    """
    out_dir = Path(base_folder) / "paper_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs_with_meta(base_folder)
    if not runs:
        raise RuntimeError("[aggregate] no runs found under base_folder")

    # extract per-run metrics
    records: List[Dict[str, Any]] = []
    for run_dir, meta in runs:
        try:
            rec = extract_run_metrics(Path(run_dir), k_last=5)
            rec.update(meta)
            records.append(rec)
        except Exception as e:
            print(f"[aggregate] skipped {run_dir}: {e}")

    if not records:
        raise RuntimeError("[aggregate] all runs failed to parse")

    raw_df = pd.DataFrame(records)
    raw_csv = out_dir / "runs_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    metrics = [
        "best_valid_loss",
        "last_valid_auc",
        "last_valid_f1",
        "test_accuracy",
    ]
    grp_df = mean_std_table(raw_df, "backbone", metrics)
    grp_csv = out_dir / "groups_mean_std.csv"
    grp_df.to_csv(grp_csv, index=False)

    seed_counts = (
        raw_df.groupby("backbone")["seed"]
        .nunique(dropna=True)
        .rename("n_seeds")
        .reset_index()
    )
    parts = [f"{r['backbone']} (n={int(r['n_seeds'])})" for _, r in seed_counts.iterrows()]
    caption = "Results: mean$\\pm$std across seeds per backbone. " \
              "Backbones: " + ", ".join(parts) + ". " \
              "Metrics computed from final epoch (valid) or test split where applicable."
    label = f"tab:results_{time.strftime('%Y%m%d_%H%M%S')}"

    tex = to_latex_pm(grp_df, "backbone", metrics, caption=caption, label=label)
    (out_dir / "table.tex").write_text(tex, encoding="utf-8")

    print(f"[aggregate] wrote:\n - {raw_csv}\n - {grp_csv}\n - {(out_dir/'table.tex')}")
    return str(out_dir)


def run_model_experiments(n_epochs: int, seed: int):
    tsai_models = [
        'RNNAttention',
        'LSTMAttention',
        'GRUAttention',
        'RNNPlus',
        'TSiT',
        'TSiTPlus',
        'ResNetPlus',
        'InceptionTimePlus',
        'XCMPlus',
        'TSSequencerPlus',
    ]

    model_status = {'successful': [], 'failed': []}
    performance_metrics: List[Dict[str, Any]] = []



    def log_error(model_name, error, tb):
        return {
            'model': model_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': ''.join(traceback.format_tb(tb))
        }

 

    for backbone_model in tsai_models:

        try:
            print(f"\nStarting experiment for model: {backbone_model} for a maximum of {n_epochs} epochs")

            hidden_size = 128
            patience = 20
            val_size = 0.2
            test_size = 0.1
            bs = 8

            features = ['revleader','new_oil_state','coldwar','majorpower','logoil','ln_gdppc','lnpop','muslim','borders','polity']
        

            args = {
                'val_size': val_size,
                'test_size': test_size,
                'hidden_size': hidden_size,
                'n_years': range(1945, 2011),
                'n_epochs': n_epochs,
                'min_epoch': min_epoch, 
                'bs': bs,
                'features': features,
                'feature_dim': len(features),
                'patience': patience,
                'device': device,
            }

            backbone_overrides = {
                'RNNAttention': {
                    'rnn_dropout': 0.3,
                    'n_heads': 8,
                    'encoder_dropout': 0.2,
                    'encoder_layers': 2,
                    'fc_dropout': 0.3,
                        },
                    # Add others here if needed later
                }
            
            args['backbone_model'] = backbone_model
            args['backbone_kwargs'] = backbone_overrides.get(backbone_model, {})

            args['variational_dropout'] = 0.3 if backbone_model == 'RNNAttention' else 0.0
            args['attn_entropy_lambda'] = 0.01 if backbone_model == 'RNNAttention' else 0.0

            folder = os.path.join(base_folder, f'epochs_{n_epochs}/{backbone_model}_val_{val_size}_test_{test_size}/')
            args['folder'] = folder
            args['model_path'] = os.path.join(folder, 'models/')
            os.makedirs(folder, exist_ok=True)
            os.makedirs(args['model_path'], exist_ok=True)
            print(f"-------------------- Created {folder} and {args['model_path']}")

            pd.DataFrame([args]).to_csv(os.path.join(folder, 'model_config.csv'), index=False)

            
            print(f"Cross-validating model: {backbone_model}")
            fold_df = run_cross_validation(args, n_splits=5, seed=42)
            print('Done cross-validation')

            def _m(c): return float(fold_df[c].mean()) if c in fold_df else float('nan')
            def _s(c): return float(fold_df[c].std())  if c in fold_df else float('nan')

            performance_metrics.append({
                'model_name': backbone_model, 'epochs': n_epochs,
                'val_size': val_size, 'test_size': test_size,
                'n_folds': len(fold_df),
                'train_accuracy': _m('train_accuracy'), 'train_accuracy_std': _s('train_accuracy'),
                'valid_accuracy': _m('valid_accuracy'), 'valid_accuracy_std': _s('valid_accuracy'),
                'test_accuracy':  _m('test_accuracy'),  'test_accuracy_std':  _s('test_accuracy'),
                'train_auc': _m('train_auc'), 'train_auc_std': _s('train_auc'),
                'valid_auc': _m('valid_auc'), 'valid_auc_std': _s('valid_auc'),
                'test_auc':  _m('test_auc'),  'test_auc_std':  _s('test_auc'),
            })

            model_status['successful'].append(backbone_model)
            print(f"✓ Successfully completed experiment for {backbone_model}")

        except Exception as e:
            _, _, tb = sys.exc_info()
            error_info = log_error(backbone_model, e, tb)
            model_status['failed'].append(error_info)
            print(f"✗ Error in experiment for {backbone_model}: {error_info['error_message']}")
            print(error_info['traceback'])

    if performance_metrics:
        performance_df = pd.DataFrame(performance_metrics).sort_values(
            by=['train_auc', 'train_accuracy'], ascending=False
        )
        print("\nMODEL PERFORMANCE SUMMARY")
        print(performance_df.to_string(index=False))
        float_cols = performance_df.select_dtypes(include=[np.number]).columns
        performance_df[float_cols] = performance_df[float_cols].round(3)

        print(performance_df.to_string(index=False))
        performance_df.to_csv(os.path.join(base_folder, f'all_models_performance_summary_{n_epochs}.csv'), index=False)

    else:
        print("No performance metrics to report. All models failed or were skipped.")

    print("\nEXECUTION SUMMARY")
    print(f"Total models processed: {len(tsai_models)}")
    print(f"Successful models: {len(model_status['successful'])}")
    print(f"Failed models: {len(model_status['failed'])}")
    if model_status['successful']:
        print("\nSuccessful models:")
        for model in model_status['successful']:
            print(f"✓ {model}")
    if model_status['failed']:
        print("\nFailed models:")
        for error in model_status['failed']:
            print(f"✗ {error['model']}: {error['error_message']}")

    try:
        reports_dir = aggregate_all_runs_for_publication(base_folder=base_folder)
        print(f"Paper-ready reports saved under: {reports_dir}")
    except Exception as e:
        print(f"[aggregate] Skipped aggregation due to error: {e}")


    return model_status


if __name__ == "__main__":
    min_epoch = 10
    n_epochs = 900
    seed = 42

    print(f"--------------- [•] PID: {os.getpid()}", flush=True)

    base_folder = '../../project_results/tsai_no_tda_Results/classification_models/'
    log_dir = os.path.join(base_folder, 'logs'); os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"run_nepochs_{n_epochs}_{ts}.log")

    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    log_file = open(log_path, mode="a", encoding="utf-8", buffering=1)

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    warnings.simplefilter("default")
    print(f"=== START {datetime.now().isoformat()} ===")
    print(f"Logging to: {log_path}")
    try:
        print("Starting model experiments...")
        summary = run_model_experiments(n_epochs,seed)
        print("\nModel experiments completed successfully.")
    except Exception:
        print("\nCritical error in main execution:")
        traceback.print_exc()
        print("\nThe program encountered an error and will terminate.")
        raise
    finally:
        print(f"=== END {datetime.now().isoformat()} ===")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        try: log_file.close()
        except Exception: pass

    try:
        reports_dir = aggregate_all_runs_for_publication(base_folder=base_folder)
        print(f"Paper-ready reports saved under: {reports_dir}")
    except Exception as e:
        print(f"[aggregate] Skipped aggregation: {e}")




