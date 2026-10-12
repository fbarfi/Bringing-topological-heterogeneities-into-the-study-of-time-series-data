
from __future__ import annotations
from functools import partial
from typing import Optional, Dict, Tuple, List
from pathlib import Path

import os, csv, io, gc
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsai.all import *
from fastai.basics import *
from fastai.learner import Learner
from fastai.callback.progress import CSVLogger
from fastai.callback.core import Callback
from fastai.metrics import Metric, accuracy
from fastai.losses import FocalLossFlat, CrossEntropyLossFlat
from fastai.optimizer import OptimWrapper
from fastai.callback.tracker import (EarlyStoppingCallback, SaveModelCallback,
                                     TrackerCallback, ReduceLROnPlateau)
from fastai.data.core import DataLoaders
from torchinfo import summary

from .preparing_data_class import PreparingDataClass
from .tsai_classifier_VAE_class import TSAIClassifier_VAE
from .making_plots import MakingPlots


# =====================================================================
# Metric: Multi-class ROC AUC (sklearn-based)
# =====================================================================
class MultiClassROCAUC(Metric):
    def __init__(self, average='macro', multi_class='ovo', num_classes=None):
        self.average = average
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.all_probs = []
        self.all_targets = []

    def reset(self):
        self.all_probs = []
        self.all_targets = []

    def accumulate(self, learn):
        logits = learn.pred
        y = learn.y if torch.is_tensor(learn.y) else learn.y[0]
        probs = torch.softmax(logits, dim=1).detach().cpu()
        self.all_probs.append(probs)
        self.all_targets.append(y.detach().cpu())

    @property
    def value(self):
        all_probs = torch.cat(self.all_probs).numpy()
        all_targets = torch.cat(self.all_targets).numpy()
        if self.num_classes is None:
            self.num_classes = len(np.unique(all_targets))
        return roc_auc_score(all_targets, all_probs,
                             multi_class=self.multi_class, average=self.average)

    def compute_roc_curves(self):
        all_probs = torch.cat(self.all_probs).numpy()
        all_targets = torch.cat(self.all_targets).numpy()
        targets_bin = label_binarize(all_targets, classes=range(self.num_classes))
        fpr, tpr = {}, {}
        all_fpr = np.unique(np.concatenate(
            [roc_curve(targets_bin[:, i], all_probs[:, i])[0]
             for i in range(self.num_classes)]
        ))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], all_probs[:, i])
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_classes
        return all_fpr, mean_tpr


# =====================================================================
# Callback: EMA of model parameters for smoother validation
# =====================================================================
class SimpleParamEMA(Callback):
    """EMA for eval: swap in smoothed weights during validation only."""
    order = 80

    def __init__(self, beta: float = 0.99, warmup_steps: int = 0):
        super().__init__()
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self.ema_step: int = 0

    def _tracked_named_params(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.data.is_floating_point():
                yield name, p

    def before_fit(self):
        self.shadow.clear()
        self._backup.clear()
        self.ema_step = 0
        for name, p in self._tracked_named_params():
            self.shadow[name] = p.data.detach().clone()

    def after_batch(self):
        if not self.training:
            return
        self.ema_step += 1
        if self.ema_step <= self.warmup_steps:
            return
        with torch.no_grad():
            for name, p in self._tracked_named_params():
                s = self.shadow.get(name)
                if s is None or s.shape != p.data.shape or s.dtype != p.data.dtype or s.device != p.data.device:
                    self.shadow[name] = p.data.detach().clone()
                else:
                    s.mul_(self.beta).add_(p.data, alpha=1.0 - self.beta)

    def before_validate(self):
        self._backup.clear()
        with torch.no_grad():
            for name, p in self._tracked_named_params():
                self._backup[name] = p.data.detach().clone()
                s = self.shadow.get(name)
                if s is not None:
                    if s.device != p.data.device or s.dtype != p.data.dtype:
                        s = s.to(device=p.data.device, dtype=p.data.dtype)
                    p.data.copy_(s)

    def after_validate(self):
        if not self._backup:
            return
        with torch.no_grad():
            for name, p in self._tracked_named_params():
                b = self._backup.get(name)
                if b is not None:
                    if b.device != p.data.device or b.dtype != p.data.dtype:
                        b = b.to(device=p.data.device, dtype=p.data.dtype)
                    p.data.copy_(b)
        self._backup.clear()


# =====================================================================
# Callback: Early stopping only after min_epoch
# =====================================================================
class EarlyStoppingAfter(EarlyStoppingCallback):
    """Early stop only after `min_epoch` (0-indexed)."""
    def __init__(self, min_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.min_epoch = int(min_epoch)

    def after_epoch(self):
        cur_epoch = getattr(self.learn, "epoch", 0)
        if cur_epoch < self.min_epoch:
            return
        return super().after_epoch()


# =====================================================================
# Callback: Save model without printing
# =====================================================================
class QuietSaveModelCallback(SaveModelCallback):
    """Drop-in replacement for SaveModelCallback that never logs to stdout."""
    order = SaveModelCallback.order

    def after_epoch(self):
        if self.every_epoch:
            if (self.epoch % self.every_epoch) == 0:
                self._save(f'{self.fname}_{self.epoch}')
        else:
            TrackerCallback.after_epoch(self)
            if self.new_best:
                self._save(f'{self.fname}')


# =====================================================================
# Callback: Per-batch VAE loss logger (buffered, no open file handles)
# =====================================================================
class VAEPerBatchBufferedLogger(Callback):
    """Buffers VAE loss decomposition per batch, writes once in after_fit."""
    order = 60

    def __init__(self, model, fname: Path):
        self.model = model
        self.fname = Path(fname)
        self._iter = 0
        self._rows: List[Dict] = []

    def __getstate__(self):
        return self.__dict__

    def before_fit(self):
        self._rows.clear()
        self._iter = 0
        self.fname.parent.mkdir(parents=True, exist_ok=True)

    def before_epoch(self):
        self._iter = 0

    def after_batch(self):
        losses: Optional[Dict] = getattr(self.model, "_last_losses", None)
        if not losses:
            return
        self._rows.append({
            "split": "train" if self.learn.training else "valid",
            "epoch": int(self.epoch), "iter": int(self._iter),
            "cls_loss": float(losses["cls_loss"].detach().cpu()),
            "vae_loss": float(losses["vae_loss"].detach().cpu()),
            "kl_loss": float(losses["kl_loss"].detach().cpu()),
            "recon_loss": float(losses["recon_loss"].detach().cpu()),
            "total_loss": float(losses["total_loss"].detach().cpu()),
            "beta": float(losses["beta"].detach().cpu()),
        })
        self._iter += 1

    def after_fit(self):
        if not self._rows:
            return
        with open(self.fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            w.writeheader()
            w.writerows(self._rows)


# =====================================================================
# Callback: GECO-style dual ascent on the conditioner-aux weight
# =====================================================================
class ConditionerDualAscent(Callback):
    """Learn the conditioner-aux weight lambda_cond by a generalization constraint.

    After each epoch, evaluate the validation split twice with the CURRENT model
    weights: once normally (conditioner on) and once with the conditioner ablated
    (model.conditioner_enabled = False, so H_back == H_back0). The difference

        violation = val_loss_on - val_loss_off        (EMA-smoothed)

    measures the conditioner's marginal effect on held-out loss. Dual ascent:

        lambda_cond <- clip( lambda_cond * exp(eta * violation), 0, lambda_max )

    When the conditioner HELPS (on < off => violation < 0), lambda_cond grows
    (more conditioner pressure). When it HURTS (violation > 0), lambda_cond
    shrinks. lambda_cond is thus never hand-set; it self-regulates so the
    conditioner is weighted only as much as keeps it from worsening
    generalization versus its own ablation.

    This ports only the dual-ascent update rule (exponential multiplier) from the
    constrained-optimization pattern; the constrained quantity here is a
    supervised held-out loss difference, not a distributional/topological match.
    """
    order = 60  # after tracker/telemetry callbacks

    def __init__(self, eta: float = 0.5, ema: float = 0.7, base_loss_fn=None):
        self.eta = float(eta)
        self.ema = float(ema)
        self._viol_ema = None
        # loss used for the on/off comparison: the plain classification loss is
        # the cleanest signal for "does the conditioner help held-out prediction".
        self.base_loss_fn = base_loss_fn if base_loss_fn is not None else CrossEntropyLossFlat()

    @torch.no_grad()
    def _val_cls_loss(self, enabled: bool) -> float:
        m = self.learn.model
        prev = getattr(m, "conditioner_enabled", True)
        m.conditioner_enabled = enabled
        was_training = m.training
        m.eval()
        total, n = 0.0, 0
        try:
            device = next(m.parameters()).device
            for batch in self.learn.dls.valid:
                # TDA dataloader yields a flat tuple: last element is the target,
                # the rest are model inputs (x, PD_pts, PD_mask, PD_dim). Move
                # each tensor to the model's device (the callback iterates the raw
                # loader, so placement isn't handled for us as in the fit loop).
                *xb, yb = batch
                xb = [t.to(device) if torch.is_tensor(t) else t for t in xb]
                yb = yb.to(device) if torch.is_tensor(yb) else yb
                pred = m(*xb)
                l = self.base_loss_fn(pred, yb)
                bs = yb.shape[0]
                total += float(l) * bs
                n += bs
        finally:
            m.conditioner_enabled = prev
            if was_training:
                m.train()
        return total / max(n, 1)

    def after_epoch(self):
        m = self.learn.model
        if not hasattr(m, "lambda_cond"):
            return
        # on-the-fly ablation reference: same weights, conditioner on vs off
        loss_on = self._val_cls_loss(enabled=True)
        loss_off = self._val_cls_loss(enabled=False)
        violation = loss_on - loss_off      # >0 => conditioner hurting held-out

        # EMA-smooth the (noisy, small-split) violation
        if self._viol_ema is None:
            self._viol_ema = violation
        else:
            self._viol_ema = self.ema * self._viol_ema + (1 - self.ema) * violation

        # dual ascent: shrink lambda when conditioner hurts, grow when it helps
        lam = float(m.lambda_cond)
        lam_new = lam * math.exp(self.eta * self._viol_ema)
        lam_new = max(0.0, min(lam_new, float(getattr(m, "cond_aux_lambda_max", 2.0))))
        m.lambda_cond.fill_(lam_new)


# =====================================================================
# Loss: Classification + VAE composite with learnable beta
# =====================================================================
class VAECompositeLossUsingModelBeta(nn.Module):
    def __init__(self, model: nn.Module, base_loss_fn: nn.Module):
        super().__init__()
        self.model = model
        self.base_loss_fn = base_loss_fn

    def forward(self, pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        cls_loss = self.base_loss_fn(pred, targ)

        mu = getattr(self.model, "_last_mu", None)
        logvar = getattr(self.model, "_last_logvar", None)
        recon = getattr(self.model, "_last_recon", None)
        x = getattr(self.model, "_last_x", None)

        # Optional TDA prior (pooled)
        p_mu = getattr(self.model, "_prior_mu", None)
        p_lv = getattr(self.model, "_prior_lv", None)

        device = pred.device
        recon_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        vae_loss = torch.tensor(0.0, device=device)

        if (mu is not None) and (logvar is not None) and (x is not None) and (recon is not None):
            recon_loss = F.mse_loss(recon, x.float())
            if (p_mu is not None) and (p_lv is not None):
                # KL(q(z|x) || p(z|H)), diagonal Gaussians
                q_var = logvar.exp()
                p_var = p_lv.exp()
                eps = 1e-6
                kl_loss = 0.5 * torch.mean(
                    (q_var + (mu - p_mu).pow(2)) / (p_var + eps)
                    - 1.0 + (p_lv - logvar)
                )
            else:
                # Fallback: KL to N(0, I)
                kl_loss = 0.5 * torch.sum(
                    torch.exp(logvar) + mu ** 2 - 1.0 - logvar, dim=1
                ).mean()
            vae_loss = recon_loss + kl_loss

        beta_val = self.model.beta.to(device) if hasattr(self.model, "beta") else torch.tensor(1.0, device=device)
        total = cls_loss + beta_val * vae_loss

        # ----- conditioner-only auxiliary loss -----
        # Force the point-conditioning attention beta to carry discriminative
        # signal by making a classification head that predicts from the pure
        # conditioner contribution alone (no backbone residual to bypass it).
        cond_aux_loss = torch.tensor(0.0, device=device)
        ca_logits = getattr(self.model, "_cond_aux_logits", None)
        lam_c = getattr(self.model, "lambda_cond", None)
        lam_c = float(lam_c) if lam_c is not None else 0.0
        if lam_c != 0.0 and ca_logits is not None:
            cond_aux_loss = lam_c * F.cross_entropy(ca_logits, targ.to(ca_logits.device))
            total = total + cond_aux_loss

        self.model._last_losses = {
            "cls_loss": cls_loss.detach(),
            "vae_loss": vae_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "cond_aux_loss": cond_aux_loss.detach(),
            "lambda_cond": torch.as_tensor(lam_c),
            "total_loss": total.detach(),
            "beta": beta_val.detach(),
        }
        return total



# =====================================================================
# Main trainer
# =====================================================================
class ModelsClass:
    def __init__(self, args: Dict):
        super().__init__()
        self.args = args
        self.set_reproducibility(seed=args.get('seed', 42))
        self.folder = args['folder']
        self.model_path = args['model_path'] 
        self.val_size = args['val_size']
        self.bs = args['bs']
        self.n_epochs = args['n_epochs']
        self.feature_names = args['features']
        self.feature_dim = args['feature_dim']
        self.device = args['device']
        self.backbone_model = args['backbone_model']
        self.patience = args['patience']
        self.min_epoch = args['min_epoch']

        prebuilt = args.get('_prebuilt', None)
        if prebuilt is not None:
            self.data_class, self.splits, self.dls = prebuilt
        else:
            self.data_class = PreparingDataClass(args)
            self.splits, self.dls = self.data_class.make_dls()

        self.seq_len = self.dls[0].dataset.X.shape[1]
        self.num_classes = len(np.unique(self.dls[0].dataset.y))
        args['seq_len'] = self.seq_len
        args['num_classes'] = self.num_classes

        self.model = TSAIClassifier_VAE(args).to(self.device)

    def set_reproducibility(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _resolve_split_dl(self, dls, split: str):
        s = split.lower()
        if isinstance(dls, DataLoaders):
            if s in ("train", "training"):
                return dls.train
            if s in ("valid", "val", "validation"):
                return dls.valid
            if s in ("test", "te"):
                if hasattr(dls, "test") and dls.test is not None:
                    return dls.test
                ds_te = getattr(self.data_class, "_ds_test", None)
                if ds_te is None:
                    raise RuntimeError("Test dataset not found; ensure make_dls() stored _ds_test.")
                dls.test = dls.test_dl(ds_te, with_labels=True, shuffle=False, device=self.device)
                return dls.test
        raise ValueError(f"Unknown split '{split}' for provided dls type {type(dls).__name__}")

    def get_class_counts(self, dls, split: str = "train") -> torch.Tensor:
        dl = self._resolve_split_dl(dls, split)
        y = getattr(dl.dataset, "y", None)
        if y is None:
            raise AttributeError("Dataset must have a `.y` tensor/array.")
        y = torch.as_tensor(y).long().view(-1).cpu()
        num_classes = int(self.num_classes) if hasattr(self, "num_classes") else int(y.max().item() + 1)
        counts = torch.bincount(y, minlength=num_classes)
        return counts.to(self.device)

    def run_model(self):
        self.dls = self.dls.to(self.device)

        params = {
            'min_delta': 0.005,
            'wd': 1.0e-2,
            'lr': 1.0e-3,
            'clip_grad': 0.1,
            'betas': (0.9, 0.98),
            'eps': 1e-8,
            'amsgrad': False,
        }

        # ---- Class weights for focal loss ----
        train_counts = self.get_class_counts(self.dls, "train").clamp_min(1)
        valid_counts = self.get_class_counts(self.dls, "valid").clamp_min(1)
        test_counts = self.get_class_counts(self.dls, "test").clamp_min(1)
        print(f'train_counts: {train_counts}, valid_counts: {valid_counts}, test_counts: {test_counts}')

        class_weights = (1.0 / train_counts)
        class_weights = (class_weights / class_weights.mean()).to(self.device, dtype=torch.float32)
        print(f'train_counts: {train_counts}, class_weights: {class_weights}')

        # ---- Optimizer ----
        opt_func = partial(OptimWrapper, opt=torch.optim.Adam,
                           betas=params['betas'], eps=params['eps'],
                           amsgrad=params['amsgrad'])

        # ---- Metrics: accuracy AND AUROC ----
        metrics = [
            accuracy,
            MultiClassROCAUC(average='weighted', multi_class='ovr',
                             num_classes=self.num_classes),
        ]

        # ---- Callbacks ----
        early_stop = EarlyStoppingAfter(
            monitor="valid_loss", min_delta=params["min_delta"],
            patience=self.patience, min_epoch=self.min_epoch,
        )
        quiet_save = QuietSaveModelCallback(
            monitor="valid_loss", comp=None, min_delta=params["min_delta"],
            fname="model", every_epoch=False, at_end=False,
            with_opt=False, reset_on_fit=True,
        )
        cb_rdp = ReduceLROnPlateau(
            monitor='valid_loss', min_delta=params['min_delta'],
            patience=max(5, self.patience // 3), min_lr=1e-6,
        )
        csv_logger = CSVLogger(fname=Path(self.folder) / 'history.csv', append=False)

        # ---- Loss: Focal + VAE composite ----
        loss_cls = FocalLossFlat(weight=class_weights)
        loss_func = VAECompositeLossUsingModelBeta(self.model, loss_cls)

        # ---- VAE batch logger ----
        vae_batch_logger = VAEPerBatchBufferedLogger(
            self.model, Path(self.folder) / "vae_history.csv",
        )

        # ---- EMA of parameters (constructed; not attached to the learner) ----
        ema_cb = SimpleParamEMA(beta=0.99)

        # Dual-ascent on the conditioner-aux weight: learns lambda_cond so the
        # conditioner is weighted only as much as keeps it from worsening held-out
        # loss versus its own ablation. Uses plain classification loss for the
        # on/off comparison.
        cond_dual = ConditionerDualAscent(
            eta=float(self.args.get('cond_dual_eta', 0.5)),
            ema=float(self.args.get('cond_dual_ema', 0.7)),
        )

        # ---- Build learner ----
        learn = Learner(
            self.dls,
            self.model,
            loss_func=loss_func,
            wd=params['wd'],
            opt_func=opt_func,
            metrics=metrics,
            cbs=[early_stop, csv_logger, vae_batch_logger, cond_dual], # quiet_save, cb_rdp, ...
        )
        learn.recorder.train_metrics = True
        # learn.recorder.valid_metrics = True
        learn.clip_grad = params["clip_grad"]

        # ---- Train ----
        with ContextManagers([learn.no_bar(), learn.no_logging()]):
            learn.fit_one_cycle(
                self.n_epochs,
                lr_max=params["lr"],
                wd=params["wd"],
                # cbs=[ema_cb],
            )
        print(f"[logs] per-batch VAE log -> {(Path(self.folder) / 'vae_history.csv').resolve()}")

        # ---- Plot training curves ----
        print("\nPlotting trained-model metrics")
        df = pd.read_csv(self.folder + 'history.csv')
        epochs = df['epoch'].values.tolist()
        df = df.drop(['epoch', 'time'], axis=1)
        n_metrics = len(df.columns)
        n_cols = 3
        n_rows = math.ceil(n_metrics / n_cols)
     
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, metric in enumerate(df.columns):
            ax = axes[i]
            ax.plot(epochs, df[metric].values, linewidth=2, label=f'{metric}')
            ax.set_title(f'{metric}', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True)

        for i in range(len(df.columns), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        os.makedirs(self.folder, exist_ok=True)
        plt.savefig(f"{self.folder}/losses_metrics.png")
        plt.close()

        # ---- Close any open file handles in callbacks ----
        to_drop = []
        for cb in list(learn.cbs):
            for attr in ("file", "_file", "_f"):
                fh = getattr(cb, attr, None)
                if isinstance(fh, io.TextIOWrapper):
                    try:
                        fh.flush()
                        fh.close()
                    except Exception:
                        pass
                    setattr(cb, attr, None)
                    to_drop.append(cb)
        if to_drop:
            learn.remove_cbs(to_drop)

        # ---- Export model ----
        PATH = Path(f"{self.folder}/models/trained_classification_model.pkl")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        learn.export(PATH)

        # ---- Save model summary ----
        summary_path = Path(f"{self.folder}/model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Model Architecture:\n")
            f.write(str(self.model) + "\n\n")

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-Trainable Parameters: {non_trainable_params:,}\n\n")

            f.write("Layer-wise parameter breakdown:\n")
            for name, param in self.model.named_parameters():
                shape_str = str(tuple(param.shape))
                f.write(f"{name:40} | Shape: {shape_str:20} | Requires Grad: {param.requires_grad}\n")

            f.write("\n\nTorchinfo Summary:\n")
            summary_logged = False
            try:
                B = 2
                T = int(self.seq_len)
                F_dim = int(self.feature_dim)

                P = None
                try:
                    dl_try = getattr(self.dls, "valid", None) or self.dls[1]
                    xb = dl_try.one_batch()
                    if isinstance(xb, (list, tuple)) and len(xb) >= 4:
                        pd_pts = xb[1]
                        if torch.is_tensor(pd_pts) and pd_pts.ndim == 4:
                            P = int(pd_pts.shape[2])
                except Exception:
                    pass
                if P is None:
                    P = 4

                dummy_ts = torch.zeros(B, T, F_dim, device=self.device, dtype=torch.float32)
                dummy_pts = torch.zeros(B, T, P, 2, device=self.device, dtype=torch.float32)
                dummy_mask = torch.zeros(B, T, P, device=self.device, dtype=torch.float32)
                dummy_dim = torch.zeros(B, T, P, device=self.device, dtype=torch.long)

                model_summary = summary(
                    self.model,
                    input_data=(dummy_ts, dummy_pts, dummy_mask, dummy_dim),
                    verbose=0,
                )
                f.write(str(model_summary))
                summary_logged = True
            except Exception as e:
                f.write(f" torchinfo.summary failed: {e}\n")
            if not summary_logged:
                f.write("Failed to run torchinfo.summary — check input shapes.\n")

        print(f"\nModel summary saved to: {summary_path}")

    def run_test(self):
        test_dls = self._resolve_split_dl(self.dls, "test")
        PATH = Path(f"{self.model_path}/trained_classification_model.pkl")
        learn = load_learner(PATH, cpu=True)

        device = self.device
        model = learn.model
        model = model.to(device)

        preds, targets = learn.get_preds(dl=test_dls)

        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()

        probs_cpu = torch.softmax(preds_cpu, dim=1)

        pred_classes = torch.argmax(probs_cpu, dim=1)
        acc = (pred_classes == targets_cpu).float().mean()
        print(f"Test Accuracy: {acc:.4f}")
        os.makedirs(self.folder, exist_ok=True)

        plt.figure(figsize=(12, 8))
        plt.plot(targets_cpu, label='True', linewidth=2, alpha=0.7)
        plt.plot(pred_classes, label='Predicted', linewidth=2, alpha=0.7)
        plt.title('True vs Predicted Classes', fontsize=14, pad=20)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.folder}/test_true_vs_predicted_classes.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        for i in range(self.num_classes):
            plt.hist(probs_cpu[:, i].numpy(), alpha=0.5, label=f'Class {i}', bins=30)
        plt.title('Prediction Probabilities Distribution by Class', fontsize=14, pad=20)
        plt.xlabel('Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.folder}/test_class_probabilities_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        self.save_classification_metrics(targets_cpu, pred_classes, acc)

    def save_classification_metrics(self, targets_cpu, pred_classes, accuracy):
        class_dict = classification_report(targets_cpu, pred_classes, digits=4, output_dict=True)
        class_df = pd.DataFrame(class_dict).transpose()
        class_df.index.name = 'Class'
        if 'accuracy' in class_df.index:
            class_df = class_df.drop('accuracy')
            cols_order = ['precision', 'recall', 'f1-score', 'support']
            class_df = class_df[cols_order]
        conf_matrix = confusion_matrix(targets_cpu, pred_classes)
        conf_df = pd.DataFrame(
            conf_matrix,
            index=[f'True_{i}' for i in range(len(conf_matrix))],
            columns=[f'Pred_{i}' for i in range(len(conf_matrix))],
        )
        conf_df.index.name = 'True Class'
        conf_df.columns.name = 'Predicted Class'
        class_df.to_csv(f"{self.folder}/classification_metrics.csv")
        conf_df.to_csv(f"{self.folder}/confusion_matrix.csv")
        overall_metrics = pd.DataFrame({'Metric': ['Accuracy'], 'Value': [f"{accuracy:.4f}"]})
        overall_metrics.to_csv(f"{self.folder}/overall_metrics.csv", index=False)
        print("\nClassification Report:")
        print(class_df)
        print("\nConfusion Matrix:")
        print(conf_df)
        print(f"\nAccuracy: {accuracy:.4f}")
