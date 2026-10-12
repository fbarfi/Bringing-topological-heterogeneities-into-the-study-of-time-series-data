
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
from fastai.metrics import Metric, accuracy, BalancedAccuracy, F1Score
from fastai.losses import FocalLossFlat, LabelSmoothingCrossEntropyFlat,BCEWithLogitsLossFlat
from fastai.optimizer import OptimWrapper
from fastai.callback.tracker import (EarlyStoppingCallback, SaveModelCallback,
                                     TrackerCallback, ReduceLROnPlateau)
from fastai.data.core import DataLoaders
from torchinfo import summary

from .preparing_data_class import PreparingDataClass
from .tsai_classifier_class import TSAIClassifierClass
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
# Main trainer
# =====================================================================
class ModelsClass:
    def __init__(self, args: Dict):
        super().__init__()
        self.args = args
        self.set_reproducibility(seed=args.get('seed', 42))
        self.folder = args['folder']
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

        self.model = TSAIClassifierClass(args).to(self.device)

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
            'min_delta': 1.0e-5,
            'wd': 1.0e-5,
            'lr': 1.0e-5,
            'clip_grad': 1.,
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
        opt_func = partial(OptimWrapper, 
                           opt=torch.optim.Adam,
                        #    betas=params['betas'], 
                        #    eps=params['eps'],
                        #    amsgrad=params['amsgrad']
                           )

        # ---- Metrics: accuracy AND AUROC ----
        metrics = [
            # BalancedAccuracy(),
                    accuracy,
                    MultiClassROCAUC(average='weighted', multi_class='ovr',
                                    num_classes=self.num_classes),
                ]

        # ---- Callbacks ----
        early_stop = EarlyStoppingAfter(
            monitor="valid_loss", 
            min_delta=params["min_delta"],
            patience=self.patience, 
            min_epoch=self.min_epoch,
        )

        # cb_save = SaveModelCallback(monitor='valid_multi_class_rocauc', comp=np.greater)

        quiet_save = QuietSaveModelCallback(
            monitor="valid_loss", comp=None, 
            min_delta=params["min_delta"],
            fname="model", 
            every_epoch=False, 
            at_end=False,
            with_opt=False, 
            reset_on_fit=True,
        )
        cb_rdp = ReduceLROnPlateau(
            monitor='valid_loss', 
            min_delta=params['min_delta'],
            patience=self.patience, 
            min_lr=params["lr"],
        )
        csv_logger = CSVLogger(fname=Path(self.folder) / 'history.csv', append=False)

        loss_func = FocalLossFlat(weight=class_weights)

        # ---- EMA of parameters, swapped in during validation ----
        ema_cb = SimpleParamEMA(beta=0.99)

        learn = Learner(
            self.dls,
            self.model,
            loss_func=loss_func,
            wd=params['wd'],
            opt_func=opt_func,
            metrics=metrics,
            cbs=[ csv_logger,  early_stop, quiet_save, cb_rdp],
        )
        learn.recorder.train_metrics = True
        # learn.recorder.valid_metrics = True
        # learn.clip_grad = params["clip_grad"]

        # ---- Train ----
        with ContextManagers([learn.no_bar(), learn.no_logging()]):
            learn.fit_one_cycle(
                self.n_epochs,
                lr_max=params["lr"],
                wd=params["wd"],
                cbs=[ema_cb],
            )

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
        PATH = Path(f"{self.folder}/models/trained_classification_model.pkl")
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
