
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import torch
import torch.nn.functional as F
from tsai.all import *
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from .preparing_data_class import PreparingDataClass, TSDatasetPDHMulti, TSDataLoaderPDHMulti
from .pd_deepsets_viz import PDDeepSetsPlots

Label_Encoder = LabelEncoder()

class MakingPlots():

    def __init__(self, args):
        super(MakingPlots, self).__init__()
        self.args = args
        self.folder = args['folder']
        self.model_path = args['model_path']
        self.bs = args['bs']

        self.feature_names = args['features']
        self.tda_features = args['tda_features']
        self.val_size = args['val_size']
        self.device = args.get('device', torch.device('cpu'))

        prebuilt = args.get('_prebuilt', None)
        if prebuilt is not None:
            self.data_class, self.splits, self.dls = prebuilt   # fold's correct split
        else:
            self.data_class = PreparingDataClass(args)
            self.splits, self.dls = self.data_class.make_dls()
        self.dls_train = self.dls[0]
        self.dls_valid = self.dls[1]
        self.dls_test  = getattr(self.dls, "test", None)
        if self.dls_test is None:
            ds_te = self.data_class._ds_test
            self.dls_test = self.dls.test_dl(ds_te, with_labels=True, bs=self.bs,
                                            shuffle=False, device=self.device)
        self.num_classes = len(torch.unique(self.dls_train.dataset.y).tolist())
        # self.viz = PDDeepSetsPlots(args)     

    # ------------ core helpers ------------
    @staticmethod
    def _model_proba(model, x, pd_pts, pd_mask, pd_dim):
        logits = model(x, pd_pts, pd_mask, pd_dim)
        return torch.softmax(logits, dim=1)

    def calculate_accuracy(self, learn, dl):
        device = self.device
        model = learn.model.to(device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, pd_pts, pd_mask, pd_dim, y in dl:
                x       = x.to(device=device, dtype=torch.float32)
                pd_pts  = pd_pts.to(device=device, dtype=torch.float32)
                pd_mask = pd_mask.to(device=device, dtype=torch.float32)
                pd_dim  = pd_dim.to(device=device, dtype=torch.long)
                y       = y.to(device=device, dtype=torch.long)
                logits  = model(x, pd_pts, pd_mask, pd_dim)
                preds   = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / max(1, total)

    # ------------ plots ------------
    def show_probas_save(self, learn, pathtosaving, figsize=(12, 6)):
        try:
            device = self.device
            model = learn.model.to(device).eval()
            all_probs = []
            with torch.no_grad():
                for x, pd_pts, pd_mask, pd_dim, _ in self.dls_test:
                    x       = x.to(device=device, dtype=torch.float32)
                    pd_pts  = pd_pts.to(device=device, dtype=torch.float32)
                    pd_mask = pd_mask.to(device=device, dtype=torch.float32)
                    pd_dim  = pd_dim.to(device=device, dtype=torch.long)
                    probs   = self._model_proba(model, x, pd_pts, pd_mask, pd_dim)
                    all_probs.append(probs.cpu())
            probs = torch.cat(all_probs, dim=0)

            plt.figure(figsize=figsize)
            for i in range(probs.shape[1]):
                plt.hist(probs[:, i].numpy(), alpha=0.5, label=f'Class {i}', bins=30)
            plt.title('Prediction Probabilities by Class')
            plt.xlabel('Probability'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
            plt.savefig(pathtosaving); plt.close()
        except Exception as e:
            print(f"Warning: Could not plot probabilities: {str(e)}")

    def plot_confusion_matrix_and_roc_auc_curve(self, learn, folder):
        device = self.device
        model = learn.model.to(device).eval()

        def _collect(dl):
            preds, targets = [], []
            with torch.no_grad():
                for x, pd_pts, pd_mask, pd_dim, y in dl:
                    x       = x.to(device=device, dtype=torch.float32)
                    pd_pts  = pd_pts.to(device=device, dtype=torch.float32)
                    pd_mask = pd_mask.to(device=device, dtype=torch.float32)
                    pd_dim  = pd_dim.to(device=device, dtype=torch.long)
                    y       = y.to(device=device, dtype=torch.long)
                    logits  = model(x, pd_pts, pd_mask, pd_dim)
                    p       = logits.argmax(dim=1)
                    preds.append(p.cpu()); targets.append(y.cpu())
            return torch.cat(preds), torch.cat(targets)

        preds_tr, y_tr = _collect(self.dls_train)
        preds_va, y_va = _collect(self.dls_valid)
        preds_te, y_te = _collect(self.dls_test)

        cm_train = confusion_matrix(y_tr.numpy(), preds_tr.numpy())
        cm_valid = confusion_matrix(y_va.numpy(), preds_va.numpy())
        cm_test  = confusion_matrix(y_te.numpy(), preds_te.numpy())

        _, axs = plt.subplots(1, 3, figsize=(15, 5)); axs = axs.flatten()
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axs[0])
        axs[0].set(xlabel='Predicted', ylabel='True', title='Train Data')
        sns.heatmap(cm_valid, annot=True, fmt='d', cmap='Blues', ax=axs[1])
        axs[1].set(xlabel='Predicted', ylabel='True', title='Validation Data')
        sns.heatmap(cm_test,  annot=True, fmt='d', cmap='Blues', ax=axs[2])
        axs[2].set(xlabel='Predicted', ylabel='True', title='Test Data')
        plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.suptitle('Confusion Matrices', y=0.98, fontsize=16)
        plt.savefig(os.path.join(folder, 'train_valid_test_confusion_matrices.png')); plt.close()

        # ---- ROC AUC (macro OVR) ----
        def _probs_targets(dl):
            probs, targets = [], []
            with torch.no_grad():
                for x, pd_pts, pd_mask, pd_dim, y in dl:
                    x       = x.to(device=device, dtype=torch.float32)
                    pd_pts  = pd_pts.to(device=device, dtype=torch.float32)
                    pd_mask = pd_mask.to(device=device, dtype=torch.float32)
                    pd_dim  = pd_dim.to(device=device, dtype=torch.long)
                    y       = y.to(device=device, dtype=torch.long)
                    p       = self._model_proba(model, x, pd_pts, pd_mask, pd_dim)
                    probs.append(p.cpu()); targets.append(y.cpu())
            return torch.cat(probs).numpy(), torch.cat(targets).numpy()

        prob_tr, y_trn = _probs_targets(self.dls_train)
        prob_va, y_val = _probs_targets(self.dls_valid)
        prob_te, y_tst = _probs_targets(self.dls_test)

        np.savez_compressed(
            os.path.join(self.folder, "fold_predictions.npz"),
            train_probs=prob_tr, train_targets=y_trn,
            valid_probs=prob_va, valid_targets=y_val,
            test_probs=prob_te,  test_targets=y_tst,
        )

        n_classes = self.num_classes
        auc_tr = roc_auc_score(y_trn, prob_tr, multi_class='ovr', average='macro')
        auc_va = roc_auc_score(y_val, prob_va, multi_class='ovr', average='macro')
        auc_te = roc_auc_score(y_tst, prob_te, multi_class='ovr', average='macro')

        yb_tr = label_binarize(y_trn, classes=list(range(n_classes)))
        yb_va = label_binarize(y_val, classes=list(range(n_classes)))
        yb_te = label_binarize(y_tst, classes=list(range(n_classes)))

        def _macro_roc(y_true_bin, probs):
            fpr = {i: None for i in range(n_classes)}
            tpr = {i: None for i in range(n_classes)}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            return all_fpr, mean_tpr

        fpr_tr, tpr_tr = _macro_roc(yb_tr, prob_tr)
        fpr_va, tpr_va = _macro_roc(yb_va, prob_va)
        fpr_te, tpr_te = _macro_roc(yb_te, prob_te)

        plt.figure(figsize=(12, 6))
        plt.plot(fpr_tr, tpr_tr, lw=2, label=f'Train Average ROC (AUC = {auc_tr:.2f})')
        plt.plot(fpr_va, tpr_va, lw=2, label=f'Valid Average ROC (AUC = {auc_va:.2f})')
        plt.plot(fpr_te, tpr_te, lw=2, label=f'Test Average ROC (AUC = {auc_te:.2f})')
        plt.plot([0,1], [0,1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Aggregate ROC Curve (Macro-averaged)'); plt.legend(loc="lower right")
        plt.savefig(os.path.join(folder, 'train_valid_test_roc_curves.png')); plt.close()

        return auc_tr, auc_va, auc_te

    # ------------ Feature importance (permutation) ------------
    def _build_dl_from_X(self, base_ds, X_new_np, bs=None, shuffle=False):
        # Reuse PD and y from base dataset
        X_np   = np.asarray(X_new_np, dtype=np.float32)
        PD     = base_ds.PD
        y_np   = base_ds.y.detach().cpu().numpy()
        dims   = getattr(base_ds, "dims", (1,2,3,4))
        ds_new = TSDatasetPDHMulti(X_np, PD, y_np, dims=dims)
        return TSDataLoaderPDHMulti(ds_new, batch_size=bs or self.bs, shuffle=shuffle, drop_last=False)

    def plot_feature_importance(self, learn, feature_names, folder):
        baseline_acc_test  = self.calculate_accuracy(learn, self.dls_test)
        baseline_acc_train = self.calculate_accuracy(learn, self.dls_train)

        ds_test  = self.dls_test.dataset
        ds_train = self.dls_train.dataset

        test_importance_scores, train_importance_scores = [], []

        for i in range(len(feature_names)):
            # test
            X_test_shuf = ds_test.X.detach().cpu().numpy().copy()
            X_test_shuf[:, :, i] = np.random.permutation(X_test_shuf[:, :, i])
            dl_test_shuf = self._build_dl_from_X(ds_test, X_test_shuf, shuffle=False)
            test_acc_shuf = self.calculate_accuracy(learn, dl_test_shuf)
            test_importance_scores.append(baseline_acc_test - test_acc_shuf)

            # train
            X_train_shuf = ds_train.X.detach().cpu().numpy().copy()
            X_train_shuf[:, :, i] = np.random.permutation(X_train_shuf[:, :, i])
            dl_train_shuf = self._build_dl_from_X(ds_train, X_train_shuf, shuffle=False)
            train_acc_shuf = self.calculate_accuracy(learn, dl_train_shuf)
            train_importance_scores.append(baseline_acc_train - train_acc_shuf)

        test_df = pd.DataFrame({'Feature': feature_names, 'Importance': test_importance_scores, 'Dataset': 'Test'})
        train_df = pd.DataFrame({'Feature': feature_names, 'Importance': train_importance_scores, 'Dataset': 'Train'})
        fi_df = pd.concat([train_df, test_df])
        fi_df.to_csv(os.path.join(folder, "feature_importance_scores.csv"), index=False)

        # plots
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        tp = train_df.sort_values('Importance', ascending=True)
        plt.barh(tp['Feature'], tp['Importance'], color='skyblue', alpha=0.8)
        plt.title('Feature Importance (Train)'); plt.xlabel('Accuracy Decrease'); plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        vp = test_df.sort_values('Importance', ascending=True)
        plt.barh(vp['Feature'], vp['Importance'], color='lightcoral', alpha=0.8)
        plt.title('Feature Importance (Test)'); plt.xlabel('Accuracy Decrease'); plt.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(folder, "feature_importance_comparison.png"), bbox_inches='tight')
        plt.close()

        fi_df.to_csv(os.path.join(folder, "train_test_feature_importance_scores.csv"), index=False)
        return fi_df

    # ------------ Feature importance (gradients + perturbations) ------------
    def calculate_feature_importance_with_gradients_and_perturbations(
        self, learn, feature_names: List[str], folder: str,
    ):
        model = learn.model.eval().to(self.device)

        # Gradient magnitude per channel over a dataloader
        def grad_importance(dl):
            Feat = dl.dataset.X.shape[2]
            acc = torch.zeros(Feat, device=self.device)
            total = 0
            for x, pd_pts, pd_mask, pd_dim, y in dl:
                x       = x.to(self.device).requires_grad_(True)
                pd_pts  = pd_pts.to(self.device)
                pd_mask = pd_mask.to(self.device)
                pd_dim  = pd_dim.to(self.device)
                y       = y.to(self.device).long()
                logits  = model(x, pd_pts, pd_mask, pd_dim)
                loss    = F.cross_entropy(logits, y)
                for p in model.parameters():  # keep model frozen w.r.t. grads accumulation
                    if p.grad is not None: p.grad = None
                if x.grad is not None: x.grad = None
                loss.backward()
                acc += x.grad.abs().sum(dim=(0,1))  # sum over (batch,time)
                total += x.shape[0] * x.shape[1]
            return (acc / max(1, total)).detach().cpu().numpy()

        grad_imp_train = grad_importance(self.dls_train)
        grad_imp_test  = grad_importance(self.dls_test)

        # Perturbation importances (reuse permutation FI)
        fi_df = self.plot_feature_importance(learn, feature_names, folder)
        test_imp  = fi_df[fi_df["Dataset"]=="Test"]["Importance"].values
        train_imp = fi_df[fi_df["Dataset"]=="Train"]["Importance"].values

        # Save combined
        out = pd.DataFrame({
            "Feature": feature_names,
            "Gradient Importance (Train)": grad_imp_train,
            "Gradient Importance (Test)":  grad_imp_test,
            "Perturbation Importance (Train)": train_imp,
            "Perturbation Importance (Test)":  test_imp,
        })
        out_path = os.path.join(folder, "feature_importance_gradients_and_perturbations.csv")
        out.to_csv(out_path, index=False)

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs[0, 0].barh(feature_names, grad_imp_train, alpha=0.7); axs[0, 0].set_title("Gradient Importance (Train)"); axs[0,0].set_xlabel("Importance"); axs[0,0].grid(alpha=0.3)
        axs[0, 1].barh(feature_names, train_imp,      alpha=0.7); axs[0, 1].set_title("Perturbation Importance (Train)"); axs[0,1].set_xlabel("Importance"); axs[0,1].grid(alpha=0.3)
        axs[1, 0].barh(feature_names, grad_imp_test,  alpha=0.7); axs[1, 0].set_title("Gradient Importance (Test)");  axs[1,0].set_xlabel("Importance"); axs[1,0].grid(alpha=0.3)
        axs[1, 1].barh(feature_names, test_imp,       alpha=0.7); axs[1, 1].set_title("Perturbation Importance (Test)");  axs[1,1].set_xlabel("Importance"); axs[1,1].grid(alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(folder, "feature_importance_gradients_and_perturbations.png")
        plt.savefig(png_path, bbox_inches="tight"); plt.close()
        return out

    # ------------ entry point ------------
    def plotting_fn(self):
        folder = self.folder
        feature_names = self.feature_names

        PATH = Path(self.model_path + f"trained_classification_model.pkl")
        learn = load_learner(PATH, cpu=True)

        # Time Series Clustering (use first channel, convert to numpy)
        X_test_np = self.dls_test.dataset.X[:, :, 0].detach().cpu().numpy()
        model_kmeans = TimeSeriesKMeans(n_clusters=self.num_classes, metric="euclidean", max_iter=100, random_state=42)
        clusters = model_kmeans.fit_predict(X_test_np)

        fig, axs = plt.subplots(4, 1, figsize=(12, 20)); axs = axs.flatten()
        for yi, ax in enumerate(axs):
            for xx in X_test_np[clusters == yi]:
                ax.plot(xx.ravel(), "k-", alpha=.2)
            ax.plot(model_kmeans.cluster_centers_[yi].ravel(), "r-")
            ax.set_title("Cluster %d" % (yi + 1))
        plt.tight_layout(); plt.savefig(os.path.join(folder, 'prediction_timeserieskmeans.png')); plt.close()

        # Accuracies
        accs = (
            self.calculate_accuracy(learn, self.dls_train),
            self.calculate_accuracy(learn, self.dls_valid),
            self.calculate_accuracy(learn, self.dls_test),
        )

        # Probabilities
        self.show_probas_save(learn, os.path.join(folder, 'prediction_probabilities.png'))

        # Confusion + ROC
        aucs = self.plot_confusion_matrix_and_roc_auc_curve(learn, folder)

        # Permutation FI
        fi_df = self.plot_feature_importance(learn, feature_names, folder)
        fi_df.to_latex(buf=os.path.join(folder, 'prediction_feature_importance.tex'), float_format="%.3f")

        # Gradients + perturbations FI
        self.calculate_feature_importance_with_gradients_and_perturbations(learn, feature_names, folder)

        # Plotting PD DeepSets illustration (conditioning attention beta)
        # Years must match the PD sequence length T (1945-2010 = 66); the viz
        # also guards against a mismatch, but keep this correct at the source.
        # years = np.arange(1945, 2011)
        # self.viz.run(which="train", batch_idx=0, years=years)
        


        return accs, aucs
