from __future__ import annotations

import os, ast, asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple, Hashable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from fastai.learner import load_learner
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .path_isolation import use_tda_code, use_base_code
from .preparing_data_class import PreparingDataClass, TSDatasetPDHMulti, TSDataLoaderPDHMulti


# =============================== config ===============================

@dataclass(frozen=True)
class CompareConfig:
    split: str = "valid"
    random_state: int = 42
    n_boot: int = 2000
    alpha: float = 0.05
    pos_label: int = 1
    default_years: Tuple[int, ...] = tuple(range(1945, 2011))


# ============================== parsing ===============================

class _ParseUtils:
    @staticmethod
    def parse_listish(v) -> list:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    lit = ast.literal_eval(s)
                    return list(lit) if isinstance(lit, (list, tuple, range)) else [str(lit)]
                except Exception:
                    return [p for p in s.strip("[]").split(",")]
            return [p for p in s.split(",")]
        return []

    @staticmethod
    def load_run_config(pkl_path: Union[str, Path]) -> Dict[str, object]:
        p = Path(pkl_path).resolve()
        run_dir = p.parent.parent
        cfg_csv = run_dir / "model_config.csv"
        if not cfg_csv.exists():
            return {}
        try:
            df = pd.read_csv(cfg_csv)
            if df.empty:
                return {}
            rec = df.iloc[0].to_dict()
            for k in ("features", "tda_features"):
                if k in rec:
                    rec[k] = _ParseUtils.parse_listish(rec[k])
            ny = rec.get("n_years", None)
            if isinstance(ny, str):
                try:
                    lit = ast.literal_eval(ny)
                    if isinstance(lit, (range, list, tuple)):
                        rec["n_years"] = list(lit)
                    else:
                        rec["n_years"] = list(range(1945, 2011))
                except Exception:
                    rec["n_years"] = list(range(1945, 2011))
            return rec
        except Exception:
            return {}


# ========================= async helpers ==============================

def _await(coro):
    return asyncio.run(coro)

def noop_activation(x):
    """Identity activation so learn.get_preds returns raw logits, not softmax."""
    return x


async def _load_learner_async(pkl_path: str):
    return await asyncio.to_thread(load_learner, pkl_path)

async def _get_preds_async(learn, dl):
    def _call():
        with learn.no_logging():
            # act=noop -> return RAW pre-activation outputs (logits), NOT softmax
            # probabilities. The calibrators (fit_on_valid / probs_from_logits)
            # and F.cross_entropy all assume logits; act=None applies softmax for
            # CrossEntropyLossFlat, which would feed probabilities to the
            # calibrators and collapse temperature fitting.
            preds, _ = learn.get_preds(dl=dl, with_decoded=False,
                                       act=noop_activation)
            return preds.detach()
    return await asyncio.to_thread(_call)


# ============================ calibrators =============================

class ClasswiseTemperatureCalibratorTorch(nn.Module):
    def __init__(self, num_classes: int, init_T: float = 1.5):
        super().__init__()
        self.log_T = nn.Parameter(torch.full((num_classes,), float(init_T)).log())

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp_min(1e-6)

    @torch.no_grad()
    def probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=1)

    def fit_on_valid(self, val_logits: torch.Tensor, val_y: torch.Tensor,
                     max_iter: int = 150, tol: float = 1e-6) -> torch.Tensor:
        opt = torch.optim.LBFGS([self.log_T], lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe")
        prev = float("inf")
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_y)
            loss.backward()
            return loss
        for _ in range(max_iter):
            loss = float(opt.step(closure))
            if abs(prev - loss) < tol:
                break
            prev = loss
        return self.T.detach().cpu()


class ClasswiseAffineCalibratorTorch(nn.Module):
    def __init__(self, num_classes: int, init_T: float = 1.5):
        super().__init__()
        self.log_T = nn.Parameter(torch.full((num_classes,), float(init_T)).log())
        self.b     = nn.Parameter(torch.zeros(num_classes))

    @property
    def T(self):
        return self.log_T.exp().clamp_min(1e-6)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T + self.b

    @torch.no_grad()
    def probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=1)

    def fit_on_valid(self, val_logits: torch.Tensor, val_y: torch.Tensor,
                     max_iter: int = 200, tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        opt = torch.optim.LBFGS(list(self.parameters()), lr=0.2, max_iter=max_iter, line_search_fn="strong_wolfe")
        prev = float("inf")
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_y)
            loss.backward()
            return loss
        for _ in range(max_iter):
            loss = float(opt.step(closure))
            if abs(prev - loss) < tol:
                break
            prev = loss
        return self.T.detach().cpu(), self.b.detach().cpu()


class TemperatureCalibratorTorch(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(init_T)).log())

    @property
    def T(self) -> torch.Tensor:
        return self.log_T.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    @torch.no_grad()
    def probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=1)

    def fit_on_valid(self, val_logits: torch.Tensor, val_y: torch.Tensor,
                     max_iter: int = 100, tol: float = 1e-6) -> float:
        opt = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")
        prev = float("inf")
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_y)
            loss.backward()
            return loss
        for _ in range(max_iter):
            loss = float(opt.step(closure))
            if abs(prev - loss) < tol:
                break
            prev = loss
        return float(self.T.item())


def _make_calibrator(mode: str, C: int):
    m = (mode or "scalar").lower()
    if m == "off":
        class _Id(nn.Module):
            @torch.no_grad()
            def probs_from_logits(self, z): return F.softmax(z, dim=1)
            def fit_on_valid(self, *a, **k): return None
        return _Id()
    if m == "vector":
        return ClasswiseTemperatureCalibratorTorch(num_classes=C)
    if m == "affine":
        return ClasswiseAffineCalibratorTorch(num_classes=C)
    return TemperatureCalibratorTorch()


def evaluate_calibration_modes(
    out_dir,
    *,
    val_logits_base: torch.Tensor,
    val_logits_tda: torch.Tensor,
    eval_logits_base: torch.Tensor,
    eval_logits_tda: torch.Tensor,
    yv_t: torch.Tensor,            # validation labels (long), already index-fixed
    y_true: np.ndarray,           # eval labels (int), already index-fixed
    C: int,
    modes=("off", "scalar", "vector", "affine"),
    n_bins: int = 15,
):
    """Refit each calibrator on val logits, apply to eval logits, recompute
    ECE + WrongPredConfidence. Reuses cached logits; no re-inference.

    Returns the per-mode DataFrame and also writes calibration_modes.csv.
    """
    # _make_calibrator and MetricsUtils are taken from the enclosing module
    # when this function is pasted into comparing_tsai_models.py.
    global _make_calibrator, MetricsUtils

    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)

    # Per-class TEST support (computed, not assumed)
    support = {int(c): int((y_true == c).sum()) for c in range(C)}
    min_support = min(support.values()) if support else 0
    n_test = int(y_true.shape[0])

    def _probs(calibrator, logits) -> np.ndarray:
        # All calibrators expose probs_from_logits; identity handles "off".
        p = calibrator.probs_from_logits(logits).detach().cpu().numpy()
        # Guard: clip + renormalize so ECE binning is well-defined.
        p = np.clip(p, 0.0, None)
        s = p.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return p / s

    rows = []
    for who, vlog, elog in (
        ("base", val_logits_base, eval_logits_base),
        ("tda",  val_logits_tda,  eval_logits_tda),
    ):
        for mode in modes:
            cal = _make_calibrator(mode, C=C)
            fitted = None
            # "off" returns None and ignores fit; others fit on validation.
            try:
                fitted = cal.fit_on_valid(vlog, yv_t)
            except Exception as e:
                # If a fit fails (e.g. degenerate class), record and skip mode.
                rows.append(dict(
                    model=who, calib_mode=mode, fit_ok=False,
                    ECE=np.nan, WrongPredConfidence=np.nan,
                    n_test=n_test, min_class_support=min_support,
                    note=f"fit_failed: {type(e).__name__}",
                ))
                continue

            p_eval = _probs(cal, elog)
            ece = MetricsUtils.ece_multiclass_macro_ovr(y_true, p_eval, n_bins=n_bins)
            wpc = MetricsUtils.wrong_pred_confidence_mean(y_true, p_eval)

            # Record the fitted temperature(s) where scalar/vector/affine.
            if isinstance(fitted, (float, int)):
                T_repr = f"{float(fitted):.4f}"
            elif isinstance(fitted, torch.Tensor):
                T_repr = ";".join(f"{v:.4f}" for v in fitted.flatten().tolist())
            elif isinstance(fitted, tuple):  # affine -> (T, b)
                T_t, T_b = fitted
                T_repr = "T=" + ";".join(f"{v:.4f}" for v in T_t.flatten().tolist()) \
                         + "|b=" + ";".join(f"{v:.4f}" for v in T_b.flatten().tolist())
            else:
                T_repr = ""  # "off"

            rows.append(dict(
                model=who, calib_mode=mode, fit_ok=True,
                ECE=float(ece), WrongPredConfidence=float(wpc),
                n_test=n_test, min_class_support=min_support,
                temperature=T_repr,
            ))

    df = pd.DataFrame(rows)
    # also attach per-class support as columns for downstream auditing
    for c in range(C):
        df[f"support_cls{c}"] = support.get(c, 0)

    out_path = os.path.join(str(out_dir), "calibration_modes.csv")
    df.to_csv(out_path, index=False)
    return df, out_path

# ============================== syncing (data) ========================

class TDAsync:
    def __init__(self, data_args: Dict[str, object]):
        self.data_args = data_args
        self.n_years   = data_args['n_years']
        self.val_size  = data_args['val_size']
        self.test_size = data_args['test_size']
        self.device    = data_args.get('device', None)
        self.split     = data_args.get('SPLIT', 'valid')

    # singleflight + rate-limiter caches
    _load_cache: Dict[str, object] = {}
    _load_locks: Dict[str, asyncio.Lock] = {}
    _pred_sems: Dict[Hashable, asyncio.Semaphore] = {}

    @staticmethod
    async def singleflight(key: Hashable, func, *args, **kwargs):
        lk = str(key)
        lock = TDAsync._load_locks.setdefault(lk, asyncio.Lock())
        async with lock:
            cached = TDAsync._load_cache.get(lk)
            if cached is not None:
                return cached
            res = await func(*args, **kwargs)
            TDAsync._load_cache[lk] = res
            return res

    @staticmethod
    async def rate_limited(key: Hashable, max_concurrent: int, func, *args, **kwargs):
        sem = TDAsync._pred_sems.setdefault(key, asyncio.Semaphore(int(max_concurrent)))
        async with sem:
            return await func(*args, **kwargs)

    @staticmethod
    def _split(y, val_size=0.2, test_size=0.1, seed=42):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        tr, va = train_test_split(tr, test_size=val_size, stratify=y[tr], random_state=seed)
        return tr, va, te

    def run(self, split: Optional[str] = None):
        """Return X, PD_list, y, used_split, years_idx (index 0..T-1)."""
        split = (split or self.split).lower()
        pdata = PreparingDataClass(self.data_args).prepare()
        X_all   = {"train": pdata["X_train"], "valid": pdata["X_valid"], "test": pdata["X_test"]}[split]
        PD_all  = {"train": pdata["PD_train"], "valid": pdata["PD_valid"], "test": pdata["PD_test"]}[split]
        y_all   = {"train": pdata["y_train"], "valid": pdata["y_valid"], "test": pdata["y_test"]}[split]
        T       = len(self.n_years)
        years_idx = np.arange(T, dtype=np.int64)  
        return X_all, PD_all, y_all, split, years_idx


# =========================== dataloaders ==============================

def _prep_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y.astype(np.int64, copy=False)

class _NumpyXYDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(_prep_labels(y), dtype=torch.long)
        assert len(self.X) == len(self.y), "X and y length mismatch"
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class _MaskedPDDatasetPDHMulti(Dataset):
    """Returns dicts compatible with PDHMulti collate; zero X[t], drop PD at t."""
    def __init__(self,
                 X: np.ndarray,
                 PD_list: List[List[Dict[int, np.ndarray]]],
                 y: np.ndarray,
                 t: int,
                 dims: Tuple[int, ...]):
        self.X = X
        self.PD = PD_list
        self.y = torch.as_tensor(_prep_labels(y), dtype=torch.long)
        self.t = int(t)
        self.dims = tuple(dims)  

    def __len__(self): 
        return int(self.y.numel())

    def __getitem__(self, idx):
        # zero X[t]
        xi = self.X[idx].copy()
        xi[self.t, :] = 0.0

        PDti = self.PD[idx]
        PD_masked = []
        for tt, dct in enumerate(PDti):
            if tt == self.t:
                PD_masked.append({})
            else:
                PD_masked.append(dct)

        clean = []
        for t_dict in PD_masked:
            t_accum = []
            for d, arr in t_dict.items():
                if arr is None: 
                    continue
                a = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
                if a.size == 0:
                    continue
                t_accum.append((int(d), a))
            clean.append(t_accum)

        return {
            "X": torch.as_tensor(xi, dtype=torch.float32),
            "PD_multi": clean,
            "y": int(self.y[idx].item()),
        }


class Dataloaders:
    @staticmethod
    def baseline(X: np.ndarray, y: np.ndarray, bs: int) -> DataLoader:
        return DataLoader(_NumpyXYDataset(X, y), batch_size=bs, shuffle=False)

    @staticmethod
    def tda(_tda_code_root: str, X: np.ndarray, PD_list: List[List[Dict[int, np.ndarray]]], y: np.ndarray, bs: int, dims: Tuple[int, ...]):
        ds = TSDatasetPDHMulti(X, PD_list, y, dims=dims)
        return TSDataLoaderPDHMulti(ds, batch_size=bs, shuffle=False, drop_last=False)

    @staticmethod
    def masked_baseline(X: np.ndarray, y: np.ndarray, t: int, bs: int) -> DataLoader:
       
        class _MaskedXYDataset(Dataset):
            def __init__(self, X, y, t):
                self.X=X; self.y=_prep_labels(y); self.t=int(t)
            def __len__(self): return len(self.y)
            def __getitem__(self, i):
                xi = self.X[i].copy(); xi[self.t,:]=0.0
                return torch.as_tensor(xi, dtype=torch.float32), torch.as_tensor(self.y[i], dtype=torch.long)
        ds = _MaskedXYDataset(X, y, t)
        return DataLoader(ds, batch_size=bs, shuffle=False)

    @staticmethod
    def masked_tda(X: np.ndarray, PD_list: List[List[Dict[int, np.ndarray]]], y: np.ndarray, t: int, bs: int, dims: Tuple[int, ...]):
        ds = _MaskedPDDatasetPDHMulti(X, PD_list, y, t=t, dims=dims)
        return TSDataLoaderPDHMulti(ds, batch_size=bs, shuffle=False, drop_last=False)


# ============================ metrics/utils ===========================

class ProbaUtils:
    @staticmethod
    def to_proba(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            a = np.stack([1 - a, a], axis=1)
        row_sum = a.sum(axis=1, keepdims=True)
        if np.all((a >= 0) & (a <= 1)) and np.allclose(row_sum, 1.0, atol=1e-5):
            return a
        z = a - a.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def pad_and_renorm(p: np.ndarray, C: int, eps: float = 1e-12) -> np.ndarray:
        if p.shape[1] == C:
            p = np.clip(p, 0.0, None)
            s = p.sum(axis=1, keepdims=True)
            zero = (s <= eps).squeeze(1)
            if zero.any():
                p[zero] = 1.0 / C
            else:
                p = p / s
            return p
        out = np.full((p.shape[0], C), eps, dtype=p.dtype)
        out[:, :p.shape[1]] = np.clip(p, 0.0, None)
        s = out.sum(axis=1, keepdims=True)
        return out / (s + eps)


class MetricsUtils:
    @staticmethod
    def ece_binary(y_true: np.ndarray, p1: np.ndarray, n_bins: int = 15) -> float:
        """One-vs-rest ECE for a single class. Bins samples by the predicted
        probability p1 of membership in this class, and within each bin compares
        the mean predicted probability (confidence) to the empirical fraction of
        true members (accuracy). This is the standard reliability-diagram ECE and
        does NOT threshold at 0.5 -- thresholding is invalid for multiclass OvR
        where per-class probabilities are routinely below 0.5."""
        y_true = np.asarray(y_true).astype(float).reshape(-1)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        # clip into [0,1] then bin; right edge inclusive for the last bin
        idx = np.clip(np.digitize(p1, bins) - 1, 0, n_bins - 1)
        ece = 0.0
        n = len(p1)
        for b in range(n_bins):
            m = idx == b
            cnt = int(m.sum())
            if cnt == 0:
                continue
            conf = p1[m].mean()                 # mean predicted prob in bin
            acc = y_true[m].mean()              # empirical fraction of positives
            ece += (cnt / n) * abs(acc - conf)
        return float(ece)

    @staticmethod
    def ece_multiclass_macro_ovr(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
        C = proba.shape[1]
        eces = []
        for c in range(C):
            yc = (y_true == c).astype(int)
            eces.append(MetricsUtils.ece_binary(yc, proba[:, c], n_bins=n_bins))
        return float(np.mean(eces))

    @staticmethod
    def softmax_entropy_mean(probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)
        return float(np.mean(ent)) if ent.size else float("nan")

    @staticmethod
    def wrong_pred_confidence_mean(y_true: np.ndarray, probs: np.ndarray) -> float:
        y_true = np.asarray(y_true).reshape(-1)
        y_hat = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        wrong = (y_hat != y_true)
        if not np.any(wrong):
            return 0.0
        return float(np.mean(conf[wrong]))

    @staticmethod
    def mean_nll_and_margin_from_proba(y_true: np.ndarray, proba: np.ndarray) -> Tuple[float, float]:
        eps = 1e-12
        idx = y_true.astype(int)
        nll = -np.log(np.clip(proba[np.arange(len(idx)), idx], eps, 1.0)).mean()
        top2 = -np.sort(-proba, axis=1)[:, :2]
        margin = np.mean(top2[:, 0] - top2[:, 1])
        return float(nll), float(margin)


# ============================== CPD core ==============================

class CPDComputer:
    """Temporal occlusion for both models; synchronized predictions via TDAsync."""
    def __init__(self, years_full: np.ndarray):
        years_full = np.asarray(years_full)
        assert years_full.ndim == 1
        self.years_full = years_full
        self.T = len(years_full)

    def data_signal_generic(self, S: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not (isinstance(S, np.ndarray) and S.ndim == 3):
            return None
        diffs_time = np.abs(np.diff(S, axis=1)).mean(axis=(0, 2))
        return self.years_full[1:], diffs_time

    def occlusion_delta_curves(
        self,
        learn_b,
        learn_t,
        Xs: np.ndarray,
        PD_s: List[List[Dict[int, np.ndarray]]],
        y_true: np.ndarray,
        C: int,
        bs: int,
        dims: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nll_base_t = np.zeros(self.T, dtype=float)
        nll_tda_t  = np.zeros(self.T, dtype=float)
        mar_base_t = np.zeros(self.T, dtype=float)
        mar_tda_t  = np.zeros(self.T, dtype=float)

        for t in range(self.T):
            # baseline (mask X[t])
            dl_b_mask = Dataloaders.masked_baseline(Xs, y_true, t, bs=bs)
            pb_logits = _await(TDAsync.rate_limited(("preds", id(learn_b)), 1, _get_preds_async, learn_b, dl_b_mask))
            pb_mask = ProbaUtils.pad_and_renorm(ProbaUtils.to_proba(pb_logits.numpy()), C)
            nll_b_t, mar_b_t = MetricsUtils.mean_nll_and_margin_from_proba(y_true, pb_mask)

            # tda (mask PD at t and zero X[t] for parity)
            dl_t_mask = Dataloaders.masked_tda(Xs, PD_s, y_true, t, bs=bs, dims=dims)
            pt_logits = _await(TDAsync.rate_limited(("preds", id(learn_t)), 1, _get_preds_async, learn_t, dl_t_mask))
            pt_mask = ProbaUtils.pad_and_renorm(ProbaUtils.to_proba(pt_logits.numpy()), C)
            nll_t_t, mar_t_t = MetricsUtils.mean_nll_and_margin_from_proba(y_true, pt_mask)

            nll_base_t[t], nll_tda_t[t] = nll_b_t, nll_t_t
            mar_base_t[t],  mar_tda_t[t] = mar_b_t,  mar_t_t

        return nll_base_t, nll_tda_t, mar_base_t, mar_tda_t


class CompareRunner:
    def __init__(self, args: Dict[str, Union[str, int, float, None]]):
        self.args = args
        self.cfg  = CompareConfig()

    def _write_series(self, out_dir: Union[str, os.PathLike], name: str,
                      years: np.ndarray, scores: np.ndarray) -> str:
        m = min(len(years), len(scores))
        out_path = Path(out_dir) / f"{name}.csv"
        pd.DataFrame({"year": np.asarray(years)[:m], "score": np.asarray(scores)[:m]}).to_csv(out_path, index=False)
        return str(out_path)

    def _write_confusion(self, out_dir: Union[str, os.PathLike], name: str,
                         y_true: np.ndarray, y_pred: np.ndarray, C: int) -> str:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
        df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(C)], columns=[f"pred_{i}" for i in range(C)])
        out_path = Path(out_dir) / f"{name}.csv"
        df.to_csv(out_path)
        return str(out_path)

    def _write_per_sample_preds(self,
                                out_dir: Union[str, os.PathLike],
                                years_idx: np.ndarray,
                                y_true: np.ndarray,
                                pb: np.ndarray,
                                pt: np.ndarray) -> str:
        y_true = np.asarray(y_true).reshape(-1)
        pb = np.asarray(pb); pt = np.asarray(pt)
        if pb.ndim != 2 or pt.ndim != 2:
            raise ValueError(f"Expected pb/pt to be 2-D, got shapes pb={pb.shape}, pt={pt.shape}")
        n = y_true.shape[0]
        if pb.shape[0] != n or pt.shape[0] != n:
            raise ValueError(f"Length mismatch: y_true={n}, pb={pb.shape[0]}, pt={pt.shape[0]}.")

        base_conf = np.max(pb, axis=1)
        tda_conf  = np.max(pt, axis=1)
        base_pred = np.argmax(pb, axis=1)
        tda_pred  = np.argmax(pt, axis=1)

        years_col = None
        if years_idx is not None:
            years_idx = np.asarray(years_idx)
            if years_idx.ndim == 0:
                years_col = np.full(n, int(years_idx))
            elif years_idx.ndim == 1 and years_idx.shape[0] == n:
                years_col = years_idx

        data = {
            "sample_idx": np.arange(n, dtype=np.int64),
            "y_true": y_true.astype(np.int64, copy=False),
            "base_pred": base_pred.astype(np.int64, copy=False),
            "tda_pred": tda_pred.astype(np.int64, copy=False),
            "agree": (base_pred == tda_pred).astype(np.int64, copy=False),
            "base_conf": base_conf.astype(np.float64, copy=False),
            "tda_conf": tda_conf.astype(np.float64, copy=False),
            "conf_delta": (tda_conf - base_conf).astype(np.float64, copy=False),
            "max_prob_delta": np.max(pt - pb, axis=1).astype(np.float64, copy=False),
        }
        # Per-class probabilities for BOTH models, so downstream calibration
        # diagnostics can compute macro-OvR (per-class) reliability/ECE rather
        # than only top-label ECE. Columns: base_p0..base_p{C-1}, tda_p0..tda_p{C-1}.
        C_cols = pb.shape[1]
        for c in range(C_cols):
            data[f"base_p{c}"] = pb[:, c].astype(np.float64, copy=False)
        for c in range(C_cols):
            data[f"tda_p{c}"] = pt[:, c].astype(np.float64, copy=False)
        if years_col is not None:
            data["years_idx"] = years_col

        df = pd.DataFrame(data)
        out_path = Path(out_dir) / "per_sample_predictions.csv"
        df.to_csv(out_path, index=False)
        return str(out_path)

    def run(self,
            base_pkl: Union[str, os.PathLike],
            tda_pkl: Union[str, os.PathLike],
            out_dir: Union[str, os.PathLike]) -> Dict[str, object]:

        base_pkl = str(base_pkl); tda_pkl = str(tda_pkl); out_dir = str(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        base_code = str(self.args.get("BASE_CODE", "") or "")
        tda_code  = str(self.args.get("TDA_CODE", "") or "")
        split_req = str(self.args.get("SPLIT", self.cfg.split))

        if not base_pkl.endswith(".pkl") or not os.path.isfile(base_pkl): raise FileNotFoundError(base_pkl)
        if not tda_pkl.endswith(".pkl")  or not os.path.isfile(tda_pkl):  raise FileNotFoundError(tda_pkl)

        # Load learners via singleflight
        with use_tda_code(tda_code):
            learn_t = _await(TDAsync.singleflight(("load", tda_pkl), _load_learner_async, tda_pkl))
        with use_base_code(base_code):
            learn_b = _await(TDAsync.singleflight(("load", base_pkl), _load_learner_async, base_pkl))

        # Configs / args
        tda_cfg  = _ParseUtils.load_run_config(tda_pkl)
        base_cfg = _ParseUtils.load_run_config(base_pkl)
        n_years  = self.args.get("n_years", tda_cfg.get("n_years", list(self.cfg.default_years)))
        bs       = int(self.args.get("bs", tda_cfg.get("bs", 8)))
        dims     = tuple(self.args.get("dims", (1,2,3,4)))

        # -------- requested split (PD-aware)
        tdas = TDAsync({**self.args, "n_years": n_years, "SPLIT": split_req})
        Xs, PDs, y_true, used_split, years_idx = tdas.run(split_req)
        dl_b = Dataloaders.baseline(Xs, y_true, bs=bs)
        dl_t = Dataloaders.tda(tda_code, Xs, PDs, y_true, bs=bs, dims=dims)

        # -------- validation split for calibration (always "valid")
        Xv, PDv, yv, _, _ = TDAsync({**self.args, "n_years": n_years, "SPLIT": "valid"}).run("valid")
        dl_b_val = Dataloaders.baseline(Xv, yv, bs=bs)
        dl_t_val = Dataloaders.tda(tda_code, Xv, PDv, yv, bs=bs, dims=dims)

        # -------- calibration
        cal_mode = str(self.args.get("CALIB", "scalar")).lower()
        val_logits_base = _await(TDAsync.rate_limited(("preds", id(learn_b)), 1, _get_preds_async, learn_b, dl_b_val))
        val_logits_tda  = _await(TDAsync.rate_limited(("preds", id(learn_t)), 1, _get_preds_async, learn_t, dl_t_val))

        Cb = val_logits_base.shape[1]
        Ct = val_logits_tda.shape[1]
        if Cb != Ct:
            raise ValueError(f"Class mismatch between models: base={Cb}, tda={Ct}")
        C = Cb

        yv = np.asarray(yv, dtype=np.int64)
        y_min, y_max = int(np.min(yv)), int(np.max(yv))
        if not (0 <= y_min and y_max < C):
            if y_min == 1 and y_max == C:
                yv = yv - 1
            else:
                mask_v = (yv >= 0) & (yv < C)
                val_logits_base = val_logits_base[mask_v]
                val_logits_tda  = val_logits_tda[mask_v]
                yv = yv[mask_v]
        yv_t = torch.as_tensor(yv, dtype=torch.long)

        cal_base = _make_calibrator(cal_mode, C=C)
        cal_tda  = _make_calibrator(cal_mode, C=C)
        Tb = cal_base.fit_on_valid(val_logits_base, yv_t)
        Tt = cal_tda.fit_on_valid(val_logits_tda,  yv_t)

        # -------- evaluate requested split
        logits_base = _await(TDAsync.rate_limited(("preds", id(learn_b)), 1, _get_preds_async, learn_b, dl_b))
        logits_tda  = _await(TDAsync.rate_limited(("preds", id(learn_t)), 1, _get_preds_async, learn_t, dl_t))
        pb = cal_base.probs_from_logits(logits_base).cpu().numpy()
        pt = cal_tda.probs_from_logits(logits_tda).cpu().numpy()

        C = max(pb.shape[1], pt.shape[1])
        pb = ProbaUtils.pad_and_renorm(pb, C)
        pt = ProbaUtils.pad_and_renorm(pt, C)

        # --- record the SAME transform applied to labels, then apply to logits ---
        # The label-range check below may MASK (drop rows) or SHIFT (-1) y_true.
        # The calibration-mode comparison needs eval logits aligned to the FINAL
        # y_true, so the raw eval logits are masked identically.
        y_true = np.asarray(y_true, dtype=np.int64)
        eval_mask = np.ones(len(y_true), dtype=bool)
        y_min, y_max = int(np.min(y_true)), int(np.max(y_true))
        if not (0 <= y_min and y_max < C):
            if y_min == 1 and y_max == C:
                y_true = y_true - 1
            else:
                eval_mask = (y_true >= 0) & (y_true < C)
                pb, pt, y_true, years_idx = pb[eval_mask], pt[eval_mask], y_true[eval_mask], np.asarray(years_idx)[eval_mask]

        # aligned raw logits for the calibration-mode comparison
        eval_logits_base = logits_base[eval_mask]
        eval_logits_tda  = logits_tda[eval_mask]

        yhat_b = np.argmax(pb, axis=1)
        yhat_t = np.argmax(pt, axis=1)

        # ---- calibration-mode comparison (reuses cached, ALIGNED logits) ----
        try:
            cal_modes_df, cal_modes_path = evaluate_calibration_modes(
                out_dir,
                val_logits_base=val_logits_base,
                val_logits_tda=val_logits_tda,
                eval_logits_base=eval_logits_base,
                eval_logits_tda=eval_logits_tda,
                yv_t=yv_t,
                y_true=y_true,          # already masked+shifted; matches eval logits
                C=C,
                n_bins=15,
            )
        except Exception as _e:
            cal_modes_path = None

        rows: List[Dict[str, object]] = []
        rows.append(dict(metric="Accuracy",
                         base=accuracy_score(y_true, yhat_b),
                         tda=accuracy_score(y_true, yhat_t),
                         delta=float(accuracy_score(y_true, yhat_t) - accuracy_score(y_true, yhat_b)),
                         higher_is_better=True))

        rows.append(dict(metric="BalancedAccuracy",
                         base=balanced_accuracy_score(y_true, yhat_b),
                         tda=balanced_accuracy_score(y_true, yhat_t),
                         delta=float(balanced_accuracy_score(y_true, yhat_t) - balanced_accuracy_score(y_true, yhat_b)),
                         higher_is_better=True))

        f1_b = f1_score(y_true, yhat_b, average="macro")
        f1_t = f1_score(y_true, yhat_t, average="macro")
        rows.append(dict(metric="F1_macro", base=f1_b, tda=f1_t, delta=float(f1_t - f1_b), higher_is_better=True))

        f1m_b = f1_score(y_true, yhat_b, average="micro")
        f1m_t = f1_score(y_true, yhat_t, average="micro")
        rows.append(dict(metric="F1_micro", base=f1m_b, tda=f1m_t, delta=float(f1m_t - f1m_b), higher_is_better=True))

        auroc_b = roc_auc_score(y_true, pb, multi_class="ovr", average="macro")
        auroc_t = roc_auc_score(y_true, pt, multi_class="ovr", average="macro")
        rows.append(dict(metric="AUROC_macro", base=auroc_b, tda=auroc_t, delta=float(auroc_t - auroc_b), higher_is_better=True))

        ap_b = float(np.mean([average_precision_score((y_true == c).astype(int), pb[:, c]) for c in range(C)]))
        ap_t = float(np.mean([average_precision_score((y_true == c).astype(int), pt[:, c]) for c in range(C)]))
        rows.append(dict(metric="AUPRC_macroOvR", base=ap_b, tda=ap_t, delta=float(ap_t - ap_b), higher_is_better=True))

        ece_b = MetricsUtils.ece_multiclass_macro_ovr(y_true, pb)
        ece_t = MetricsUtils.ece_multiclass_macro_ovr(y_true, pt)
        rows.append(dict(metric="ECE_macroOvR", base=ece_b, tda=ece_t, delta=float(ece_t - ece_b), higher_is_better=False))

        wpc_b = MetricsUtils.wrong_pred_confidence_mean(y_true, pb)
        wpc_t = MetricsUtils.wrong_pred_confidence_mean(y_true, pt)
        rows.append(dict(metric="WrongPredConfidence", base=wpc_b, tda=wpc_t, delta=float(wpc_t - wpc_b), higher_is_better=False))

        summary = pd.DataFrame(rows)
        sum_path = os.path.join(out_dir, "summary.csv")
        summary.to_csv(sum_path, index=False)

        per_cls = []
        for c in range(C):
            yc = (y_true == c).astype(int)
            au_b = roc_auc_score(yc, pb[:, c]) if (yc.sum() and (1 - yc).sum()) else np.nan
            au_t = roc_auc_score(yc, pt[:, c]) if (yc.sum() and (1 - yc).sum()) else np.nan
            apb  = average_precision_score(yc, pb[:, c]) if yc.sum() else np.nan
            apt  = average_precision_score(yc, pt[:, c]) if yc.sum() else np.nan
            per_cls.append(dict(
                cls=c,
                AUROC_base=au_b, AUROC_tda=au_t, AUROC_delta=(au_t - au_b) if (not np.isnan(au_b) and not np.isnan(au_t)) else np.nan,
                AUPRC_base=apb,  AUPRC_tda=apt, AUPRC_delta=(apt - apb) if (not np.isnan(apb) and not np.isnan(apt)) else np.nan,
                support=int(yc.sum())
            ))
        per_cls_df = pd.DataFrame(per_cls)
        per_cls_path = os.path.join(out_dir, "per_class_auroc_auprc.csv")
        per_cls_df.to_csv(per_cls_path, index=False)

        per_sample_path = self._write_per_sample_preds(out_dir, years_idx, y_true, pb, pt)

        cm_base_path = self._write_confusion(out_dir, "confusion_base", y_true, np.argmax(pb, axis=1), C)
        cm_tda_path  = self._write_confusion(out_dir, "confusion_tda",  y_true, np.argmax(pt, axis=1), C)

        # ================== CPD (time-axis via occlusion) ===================
        years_full = np.asarray(n_years)
        if years_full.ndim != 1:
            raise ValueError("n_years must be a 1-D calendar vector (length T)")
        cpd = CPDComputer(years_full)

        base_signal_path = None
        tda_signal_path  = None
        sig_base = cpd.data_signal_generic(Xs)
        if sig_base is not None:
            yrs_sig, diffs_time = sig_base
            base_signal_path = self._write_series(out_dir, "cpd_base_signal", yrs_sig, diffs_time)

        nll_base_t, nll_tda_t, mar_base_t, mar_tda_t = cpd.occlusion_delta_curves(
            learn_b, learn_t, Xs, PDs, y_true, C=C, bs=bs, dims=dims
        )
        delta_nll_t    = nll_tda_t - nll_base_t
        delta_margin_t = mar_tda_t - mar_base_t

        cpd_delta_nll_path    = self._write_series(out_dir, "cpd_delta_model_nll",    years_full, delta_nll_t)
        cpd_delta_margin_path = self._write_series(out_dir, "cpd_delta_model_margin", years_full, delta_margin_t)

        self._write_series(out_dir, "cpd_base_model_nll_time",    years_full, nll_base_t)
        self._write_series(out_dir, "cpd_tda_model_nll_time",     years_full, nll_tda_t)
        self._write_series(out_dir, "cpd_base_model_margin_time", years_full, mar_base_t)
        self._write_series(out_dir, "cpd_tda_model_margin_time",  years_full, mar_tda_t)

        # Persist scalar temperatures (if scalar mode)
        try:
            if isinstance(Tb, (float, int)) and isinstance(Tt, (float, int)):
                pd.DataFrame({"model": ["base", "tda"], "T": [float(Tb), float(Tt)]}) \
                  .to_csv(os.path.join(out_dir, "temperatures.csv"), index=False)
        except Exception:
            pass

        return {
            "summary_csv": sum_path,
            "per_class_csv": per_cls_path,
            "per_sample_csv": per_sample_path,
            "confusion_csvs": {"base": cm_base_path, "tda": cm_tda_path},
            "figures": {},
            "used_split": used_split,
            "cpd_csvs": {
                "cpd_base_signal": base_signal_path,
                "cpd_delta_model_nll":    cpd_delta_nll_path,
                "cpd_delta_model_margin": cpd_delta_margin_path,
            },
            "temperatures_csv": os.path.join(out_dir, "temperatures.csv"),
            "calibration_modes_csv": cal_modes_path,
        }

# ========================= functional API ============================

def compare_with_args(
    args: Dict[str, Union[str, int, float, None]],
    base_pkl: Union[str, os.PathLike],
    tda_pkl: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
) -> Dict[str, object]:
    return CompareRunner(args).run(base_pkl=base_pkl, tda_pkl=tda_pkl, out_dir=out_dir)
