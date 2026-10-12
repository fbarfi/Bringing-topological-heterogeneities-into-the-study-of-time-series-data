
import os
from pathlib import Path
from tsai.all import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import torch
import torch.nn.functional as F

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import confusion_matrix,  roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
Label_Encoder = LabelEncoder()

from .preparing_data_class import PreparingDataClass
from .base_dataloader import TSDataLoaderClass

class MakingPlots():

    def __init__(self, args):
        super(MakingPlots, self).__init__()
        self.args = args
        self.folder = args['folder']
        self.model_path = args['model_path']
        self.bs = args['bs']

        self.feature_names = args['features']
        self.val_size = args['val_size']
        self.device = args.get('device', torch.device('cpu'))

        # self.data_class = PreparingDataClass(args)
        # self.data_dict = self.data_class.train_valid_split_data()
        # self.splits, self.dls = self.data_class.make_dls()
        # self.dls_train = self.dls[0]
        # self.dls_valid = self.dls[1]
        # self.dls_test  = getattr(self.dls, "test", None)
        # if self.dls_test is None:
        #     ds_te = self.data_class._ds_test  # set in make_dls()
        #     self.dls_test = self.dls.test_dl(ds_te, with_labels=True, bs=self.bs, shuffle=False, device=self.device)
        # self.num_classes = len(np.unique(self.dls[0].dataset.y))
       
        prebuilt = args.get('_prebuilt', None)
        if prebuilt is not None:
            # use the FOLD's data_class + dls (correct held-out test set)
            self.data_class, self.splits, self.dls = prebuilt
        else:
            # single train/valid/test split (no CV fold supplied)
            self.data_class = PreparingDataClass(args)
            self.splits, self.dls = self.data_class.make_dls()
        self.dls_train = self.dls[0]
        self.dls_valid = self.dls[1]
        self.dls_test  = getattr(self.dls, "test", None)
        if self.dls_test is None:
            ds_te = self.data_class._ds_test
            self.dls_test = self.dls.test_dl(ds_te, with_labels=True, bs=self.bs,
                                            shuffle=False, device=self.device)
        self.num_classes = len(np.unique(self.dls[0].dataset.y))

    def show_probas_save(self, learn, pathtosaving, figsize=(12, 6)):
        """Plot prediction probabilities for each class."""
        try:
            device = self.device

            all_probs = []
            model = learn.model
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                for X_batch, _ in self.dls_test:
                    X_batch = X_batch.to(dtype=torch.float32).to(device)

                    outputs = model(X_batch)
                    probs = torch.softmax(outputs, dim=1)

                    all_probs.append(probs.cpu())

            probs = torch.cat(all_probs, dim=0)

            plt.figure(figsize=figsize)
            for i in range(probs.shape[1]):
                plt.hist(probs[:, i].numpy(),
                        alpha=0.5,
                        label=f'Class {i}',
                        bins=30)

            plt.title('Prediction Probabilities by Class')
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(pathtosaving)
            plt.close()

        except Exception as e:
            print(f"Warning: Could not plot probabilities: {str(e)}")

    def plot_confusion_matrix_and_roc_auc_curve(self, learn, folder):
        """Plot confusion matrix."""
        device = self.device

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in self.dls_train:
                X_batch = X_batch.to(device)
                outputs = learn.model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        cm_train = confusion_matrix(targets, preds)

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in self.dls_valid:
                X_batch = X_batch.to(device)
                outputs = learn.model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        cm_valid = confusion_matrix(targets, preds)

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in self.dls_test:
                X_batch = X_batch.to(device)
                outputs = learn.model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        cm_test = confusion_matrix(targets, preds)

        _, axs = plt.subplots(1,3,figsize=(15, 5))
        axs = axs.flatten()

        ax = axs[0]
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Train Data')

        ax = axs[1]
        sns.heatmap(cm_valid, annot=True, fmt='d', cmap='Blues',ax= ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Validation Data')

        ax = axs[2]
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',ax= ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Test Data')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.suptitle('Confusion Matrices', y=0.98, fontsize=16)
        plt.savefig(folder + 'train_valid_test_confusion_matrices.png')
        plt.close()

        """Plot ROC curves and compute AUC for train and validation sets."""

        def get_predictions(dataloader):
            all_probs = []
            all_targets = []
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(device)

                    outputs = learn.model(X_batch)
                    probs = torch.nn.functional.softmax(outputs, dim=1)

                    all_probs.append(probs.cpu())
                    all_targets.append(y_batch.cpu())

           

            return torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy()

        train_probs, train_targets = get_predictions(self.dls_train)
        valid_probs, valid_targets = get_predictions(self.dls_valid)
        test_probs, test_targets = get_predictions(self.dls_test)

        np.savez_compressed(
                os.path.join(self.folder, "fold_predictions.npz"),
                train_probs=train_probs, train_targets=train_targets,
                valid_probs=valid_probs, valid_targets=valid_targets,
                test_probs=test_probs,   test_targets=test_targets,
            )
        
       
        train_auc = roc_auc_score(train_targets, train_probs, multi_class='ovr', average='macro')
        valid_auc = roc_auc_score(valid_targets, valid_probs, multi_class='ovr', average='macro')
        test_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')

        n_classes = self.num_classes
        train_targets_bin = label_binarize(train_targets, classes=range(n_classes))
        valid_targets_bin = label_binarize(valid_targets, classes=range(n_classes))
        test_targets_bin = label_binarize(test_targets, classes=range(n_classes))

        fpr_train = dict()
        tpr_train = dict()
        fpr_valid = dict()
        tpr_valid = dict()
        fpr_test= dict()
        tpr_test = dict()

        for i in range(n_classes):
            fpr_train[i], tpr_train[i], _ = roc_curve(train_targets_bin[:, i], train_probs[:, i])
            fpr_valid[i], tpr_valid[i], _ = roc_curve(valid_targets_bin[:, i], valid_probs[:, i])
            fpr_test[i], tpr_test[i], _ = roc_curve(test_targets_bin[:, i], test_probs[:, i])

        all_fpr_train = np.unique(np.concatenate([fpr_train[i] for i in range(n_classes)]))
        all_fpr_valid = np.unique(np.concatenate([fpr_valid[i] for i in range(n_classes)]))
        all_fpr_test = np.unique(np.concatenate([fpr_test[i] for i in range(n_classes)]))

        mean_tpr_train = np.zeros_like(all_fpr_train)
        mean_tpr_valid = np.zeros_like(all_fpr_valid)
        mean_tpr_test = np.zeros_like(all_fpr_test)

        for i in range(n_classes):
            mean_tpr_train += np.interp(all_fpr_train, fpr_train[i], tpr_train[i])
            mean_tpr_valid += np.interp(all_fpr_valid, fpr_valid[i], tpr_valid[i])
            mean_tpr_test += np.interp(all_fpr_test, fpr_test[i], tpr_test[i])

        mean_tpr_train /= n_classes
        mean_tpr_valid /= n_classes
        mean_tpr_test /= n_classes

        plt.figure(figsize=(12, 6))

        plt.plot(all_fpr_train, mean_tpr_train, color='blue', lw=2,
                label=f'Train Average ROC (AUC = {train_auc:.2f})')
        plt.plot(all_fpr_valid, mean_tpr_valid, color='red', lw=2,
                label=f'Valid Average ROC (AUC = {valid_auc:.2f})')
        plt.plot(all_fpr_test, mean_tpr_test, color='green', lw=2,
                label=f'Test Average ROC (AUC = {test_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Aggregate ROC Curve (Macro-averaged)')
        plt.legend(loc="lower right")

        plt.savefig(folder + 'train_valid_test_roc_curves.png')
        plt.close()

        return train_auc, valid_auc, test_auc


    def plot_feature_importance(self, learn, feature_names, folder):

        # Baseline accuracies
        baseline_acc_test = self.calculate_accuracy(learn, self.dls_test)
        baseline_acc_train = self.calculate_accuracy(learn, self.dls_train)


        y_test = self.dls_test.dataset.y

        y_train = self.dls_train.dataset.y

        test_importance_scores = []
        train_importance_scores = []

        for i in range(len(feature_names)):
            # Shuffle test data for feature i
            X_test_shuffled = self.dls_test.dataset.X.detach().numpy().copy()
            X_test_shuffled[:, :, i] = np.random.permutation(X_test_shuffled[:, :, i])  # Shuffle feature i

            dls_test_shuffled = TSDataLoaderClass(
                X_test_shuffled,  # Pass full X_test with shuffled feature i
                y_test,
                bs=self.bs,
                shuffle=False
            )

            # Shuffle train data for feature i
            X_train_shuffled = self.dls_train.dataset.X.detach().numpy().copy()
            X_train_shuffled[:, :, i] = np.random.permutation(X_train_shuffled[:, :, i])  # Shuffle feature i

            dls_train_shuffled = TSDataLoaderClass(
                X_train_shuffled,  # Pass full X_train with shuffled feature i
                y_train,
                bs=self.bs,
                shuffle=False
            )

            test_shuffled_acc = self.calculate_accuracy(learn, dls_test_shuffled)
            test_importance = baseline_acc_test - test_shuffled_acc
            test_importance_scores.append(test_importance)

            train_shuffled_acc = self.calculate_accuracy(learn, dls_train_shuffled)
            train_importance = baseline_acc_train - train_shuffled_acc
            train_importance_scores.append(train_importance)

        test_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': test_importance_scores,
            'Dataset': 'Test'
        })

        train_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': train_importance_scores,
            'Dataset': 'Train'
        })

        fi_df = pd.concat([train_df, test_df])

        fi_df.to_csv(f"{folder}/feature_importance_scores.csv", index=False)


        test_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': test_importance_scores,
            'Dataset': np.repeat('Test',len(feature_names)),
        })

        train_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': train_importance_scores,
            'Dataset': np.repeat('Train',len(feature_names)),
        })

        fi_df = pd.concat([train_df, test_df])

        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        train_plot = train_df.sort_values('Importance', ascending=True)
        plt.barh(train_plot['Feature'], train_plot['Importance'],
                color='skyblue', alpha=0.8)
        plt.title('Feature Importance (Train)', fontsize=12, pad=15)
        plt.xlabel('Accuracy Decrease', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        test_plot = test_df.sort_values('Importance', ascending=True)
        plt.barh(test_plot['Feature'], test_plot['Importance'],
                color='lightcoral', alpha=0.8)
        plt.title('Feature Importance (Test)', fontsize=12, pad=15)
        plt.xlabel('Accuracy Decrease', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.savefig(folder + "feature_importance_comparison.png",
                    bbox_inches='tight')
        plt.close()

        fi_df.to_csv(folder + "train_test_feature_importance_scores.csv", index=False)

        return fi_df



    def calculate_accuracy(self, learn, dl):
        """Helper function to calculate accuracy."""
        device = self.device
        correct = 0
        total = 0

        model = learn.model
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for X_batch, y_batch  in dl:

                X_batch = X_batch.to(dtype=torch.float32).to(device)
                y_batch = y_batch.to(dtype=torch.float32).to(device)

                outputs = model(X_batch)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()

                total += y_batch.size(0)

        accuracy = correct / total
        return accuracy


    def calculate_feature_importance_with_gradients_and_perturbations(
        self,
        learn,
        feature_names: List[str],
        folder: str,
    ):
        """
        Compute per-feature importance two ways:
        1) Gradient magnitude wrt inputs (per channel).
        2) Perturbation drop in accuracy by zeroing one channel.

        Notes:
        - Assumes X shape (N, T, F). We average over (N, T) to get per-channel.
        - Ensures y dtype long, shape (N,), and all tensors on same device.
        - Freezes model params so only X accumulates grads.
        """

        model = learn.model
        model.eval()
        device = next(model.parameters()).device

        # Freeze model params; we only want gradients wrt inputs.
        for p in model.parameters():
            p.requires_grad_(False)  

        # Pull datasets and move to device with correct dtypes
        # Expect shapes: X: (N, T, F); y: (N,) 
        def _prep_split(dls_split):
            ds = dls_split.dataset
            X = ds.X.detach().clone().to(device).requires_grad_(True)
            y = ds.y.detach().clone().to(device)
            if y.dim() > 1:
                y = y.view(-1)
            y = y.long()  
           
            return X, y

        X_train, y_train = _prep_split(self.dls_train)
        X_test,  y_test  = _prep_split(self.dls_test)

        # Baseline accuracies (unperturbed)
        baseline_acc_test  = self.calculate_accuracy(learn, self.dls_test)
        baseline_acc_train = self.calculate_accuracy(learn, self.dls_train)

        test_importance_scores = []
        train_importance_scores = []

        def _forward(x):
            out = model(x) 
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        Feat = X_train.shape[2]
        assert Feat == len(feature_names), (
            f"feature_names length ({len(feature_names)}) != channels ({Feat})."
        )

        # --- GRADIENT-BASED IMPORTANCE (computed once per split) ---
        # Test gradients
        if X_test.grad is not None:
            X_test.grad.zero_()
        logits_test = _forward(X_test)
        if logits_test.device != y_test.device:
            y_test = y_test.to(logits_test.device)
        loss_test = F.cross_entropy(logits_test, y_test)
        loss_test.backward()
        # Mean abs gradient over (batch, time) -> per-channel
        grad_imp_test = X_test.grad.abs().mean(dim=(0, 2)).detach().cpu()

        # Train gradients
        if X_train.grad is not None:
            X_train.grad.zero_()
        logits_train = _forward(X_train)
        if logits_train.device != y_train.device:
            y_train = y_train.to(logits_train.device)
        loss_train = F.cross_entropy(logits_train, y_train)
        loss_train.backward()
        grad_imp_train = X_train.grad.abs().mean(dim=(0, 2)).detach().cpu()

        # --- PERTURBATION-BASED IMPORTANCE ---
        # Rebuild loaders per feature with zeroed channel i.
        for i, feature_name in enumerate(feature_names):
            # TEST perturbation
            X_test_pert = X_test.detach().clone()
            X_test_pert[:, :, i] = 0
            dl_test_pert = TSDataLoaderClass(
                X_test_pert, y_test, bs=self.bs, shuffle=False
            )
            test_perturbed_acc = self.calculate_accuracy(learn, dl_test_pert)
            test_pert_imp = baseline_acc_test - test_perturbed_acc

            # TRAIN perturbation
            X_train_pert = X_train.detach().clone()
            X_train_pert[:, :, i] = 0
            dl_train_pert = TSDataLoaderClass(
                X_train_pert, y_train, bs=self.bs, shuffle=False
            )
            train_perturbed_acc = self.calculate_accuracy(learn, dl_train_pert)
            train_pert_imp = baseline_acc_train - train_perturbed_acc

            # Collect rows
            test_importance_scores.append({
                "Feature": feature_name,
                "Gradient Importance": float(grad_imp_test[i]),
                "Perturbation Importance": float(test_pert_imp),
                "Dataset": "Test",
            })
            train_importance_scores.append({
                "Feature": feature_name,
                "Gradient Importance": float(grad_imp_train[i]),
                "Perturbation Importance": float(train_pert_imp),
                "Dataset": "Train",
            })

        importance_scores = pd.DataFrame(test_importance_scores + train_importance_scores)
        csv_path = os.path.join(folder, "feature_importance_gradients_and_perturbations.csv")
        importance_scores.to_csv(csv_path, index=False)

        # ---- PLOTS ----
        train_scores = importance_scores[importance_scores["Dataset"] == "Train"]
        test_scores  = importance_scores[importance_scores["Dataset"] == "Test"]

        train_gradients = train_scores["Gradient Importance"].values
        train_perturbations = train_scores["Perturbation Importance"].values
        test_gradients  = test_scores["Gradient Importance"].values
        test_perturbations = test_scores["Perturbation Importance"].values

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        axs[0, 0].barh(feature_names, train_gradients, alpha=0.7)
        axs[0, 0].set_title("Gradient Importance (Train)")
        axs[0, 0].set_xlabel("Importance Score")
        axs[0, 0].grid(alpha=0.3)

        axs[0, 1].barh(feature_names, train_perturbations, alpha=0.7)
        axs[0, 1].set_title("Perturbation Importance (Train)")
        axs[0, 1].set_xlabel("Importance Score")
        axs[0, 1].grid(alpha=0.3)

        axs[1, 0].barh(feature_names, test_gradients, alpha=0.7)
        axs[1, 0].set_title("Gradient Importance (Test)")
        axs[1, 0].set_xlabel("Importance Score")
        axs[1, 0].grid(alpha=0.3)

        axs[1, 1].barh(feature_names, test_perturbations, alpha=0.7)
        axs[1, 1].set_title("Perturbation Importance (Test)")
        axs[1, 1].set_xlabel("Importance Score")
        axs[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(folder, "feature_importance_gradients_and_perturbations.png")
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        # Unfreeze model params back (good hygiene if caller continues training later)
        for p in model.parameters():
            p.requires_grad_(True)

        return importance_scores


    def plotting_fn(self):

        folder = self.folder
        feature_names = self.feature_names

        PATH = Path(self.model_path + f"trained_classification_model.pkl")
        learn = load_learner(PATH, cpu=True)

        """Time Series Clustering"""

        model_kmeans = TimeSeriesKMeans(n_clusters=self.num_classes,
                                      metric="euclidean",
                                      max_iter=100,
                                      random_state=42)

        clusters = model_kmeans.fit_predict(self.dls_test.dataset.X[:, :, 0])

        fig, axs = plt.subplots(4, 1, figsize=(12, 20))
        axs = axs.flatten()
        for yi, ax in enumerate(axs):
            for xx in self.dls_test.dataset.X[clusters == yi, :, 0]:
                ax.plot(xx.ravel(), "k-", alpha=.2)
            ax.plot(model_kmeans.cluster_centers_[yi].ravel(), "r-")
            ax.set_title("Cluster %d" % (yi + 1))
        plt.tight_layout()
        plt.savefig(folder + 'prediction_timeserieskmeans.png')
        plt.close()

        """ Calculate train, valid, test accuracies """

        accs = (
            self.calculate_accuracy(learn, self.dls_train),
            self.calculate_accuracy(learn, self.dls_valid),
            self.calculate_accuracy(learn, self.dls_test),
        )

        """Plotting probabilities"""
        self.show_probas_save(learn, folder + 'prediction_probabilities.png')


        """Confusion Matrix and roc-auc curve """
        aucs = self.plot_confusion_matrix_and_roc_auc_curve(learn, folder)

        """Feature importance"""
        fi_df = self.plot_feature_importance(learn, feature_names, folder)
        
        fi_df.to_latex(buf=folder + 'prediction_feature_importance.tex', float_format="%.3f")

        """ feature importance using gradients and perturbations """
        self.calculate_feature_importance_with_gradients_and_perturbations(learn, feature_names, folder)

        return accs,aucs
