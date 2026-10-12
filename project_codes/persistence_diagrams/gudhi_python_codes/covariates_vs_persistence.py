import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance

from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Tuple


class CovariatePersistencePipeline:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.homology_dimensions = args['homology_dimensions']
        self.Hidx = int(''.join(map(str, self.homology_dimensions[1:])))
        self.persistence_diagrams_path = args["PersistenceDiagrams"]
        self.metadata_path = os.path.join(args["folder"], "embedding_metadata.json")
        self.covariate_source = args.get("covariate_csv", "../../project_data/dataset_augm.csv")
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        self.year_index_to_actual = {i: year_block["year"] for i, year_block in enumerate(metadata["n_years"])}


    ###########################################################################
    # MAIN PIPELINE
    ###########################################################################
    def run_pipeline(self):
        """
        Run the full ML pipeline separately for H1, H2, H3, H4.
        Produces 4 plots: one per homology dimension.
        """
        print(f"[INFO] Running full pipeline for H1–H4")

        for h_dim in self.homology_dimensions[1:]:
            print(f"\n[INFO] >>> Processing H{h_dim}")
            self.h_dim = h_dim

            pd_map = self.load_persistence_diagrams(h_dim=h_dim)
            if not pd_map:
                print(f"[WARN] No persistence diagrams found for H{h_dim}")
                continue

            cov_df = self.load_covariates()

            df_pairs = self.build_pairwise_dataset(pd_map, cov_df)

            if df_pairs.empty:
                print(f"[WARN] Empty pairwise dataset for H{h_dim}, skipping.")
                continue

            # Train two models: Wasserstein and Bottleneck
            model_wass, feat_wass = self.train_model(df_pairs, target_col="wasserstein")
            model_bottle, feat_bottle = self.train_model(df_pairs, target_col="bottleneck")

            wasserstein_feat = pd.Series(model_wass.feature_importances_, index=feat_wass)
            bottleneck_feat = pd.Series(model_bottle.feature_importances_, index=feat_bottle)

            # Make the comparison plot for this H-dimension
            self.plot_feature_importance_comparison(wasserstein_feat, bottleneck_feat, h_dim)

    ###########################################################################
    # DATA LOADING
    ###########################################################################
    def load_persistence_diagrams(self, h_dim: int = 1) -> Dict[int, np.ndarray]:
        """
        Load persistence diagrams per homology dimension.
        """
        pd_map = {}
        for fname in os.listdir(self.persistence_diagrams_path):
            if not fname.startswith(f"PD_H_{self.Hidx}_") or not fname.endswith(".csv"):
                continue

            year_idx = int(fname.split("_")[-1].replace(".csv", ""))
            df = pd.read_csv(os.path.join(self.persistence_diagrams_path, fname))

            diag = df[df["h_dim"] == h_dim][["b", "d"]].values

            if diag.size > 0:
                pd_map[year_idx] = diag

        return pd_map

    def load_covariates(self) -> pd.DataFrame:
        """Load annual covariates and average over states."""
        df = pd.read_csv(self.covariate_source).sort_values(by=["year", "ccode"]).reset_index(drop=True)
        all_covs = self.args["all_covariates"]

        df = df[["year"] + all_covs]
        return df.groupby("year").mean()

    ###########################################################################
    # PAIRWISE DATASET
    ###########################################################################
    def build_pairwise_dataset(self, 
                           pd_map: Dict[int, np.ndarray], 
                           cov_df: pd.DataFrame,
                           ) -> pd.DataFrame:
        rows = []

        # ✅ Only include years with non-empty PDs
        valid_years = [y for y, diag in pd_map.items() if diag.size > 0]

        for i, idx1 in enumerate(valid_years):
            for idx2 in valid_years[i + 1:]:
                y1 = self.year_index_to_actual.get(idx1)
                y2 = self.year_index_to_actual.get(idx2)

                if y1 not in cov_df.index or y2 not in cov_df.index:
                    print(f"[WARN] Skipping year pair ({y1}, {y2}) due to missing covariates.")
                    continue

                dgm1 = pd_map[idx1]
                dgm2 = pd_map[idx2]


                # Compute distances
                wass = wasserstein_distance(dgm1, dgm2, order=2.) 
                bott = gd.bottleneck_distance(dgm1, dgm2)

                cov1 = cov_df.loc[y1].values
                cov2 = cov_df.loc[y2].values
                delta = cov2 - cov1

                row = {
                    **{f"delta_{k}": v for k, v in zip(cov_df.columns, delta)},
                    "year1": y1,
                    "year2": y2,
                    "wasserstein": wass,
                    "bottleneck": bott,
                }
                rows.append(row)

        if not rows:
            print(f"[WARN] No valid year pairs for H{self.h_dim}. Returning empty dataset.")
            return pd.DataFrame()

        return pd.DataFrame(rows)

    ###########################################################################
    # TRAINING
    ###########################################################################
    def train_model(self, df: pd.DataFrame, target_col: str):
        """
        Train random forest on selected persistence distance type.
        """
        X = df[[c for c in df.columns if c.startswith("delta_")]]
        y = df[target_col]

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X, y)

        return model, X.columns.tolist()

    ###########################################################################
    # PLOTTING
    ###########################################################################

    def plot_feature_importance_comparison(
        self,
        wasserstein_feat: pd.Series,
        bottleneck_feat: pd.Series,
        h_dim: int
    ):
        df_plot = pd.DataFrame({
            "Variable": [feat.replace("delta_", "") for feat in wasserstein_feat.index],
            "Wasserstein": wasserstein_feat.values,
            "Bottleneck": bottleneck_feat.reindex(wasserstein_feat.index).values
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # ----------- WASSERSTEIN ------------
        df_w = df_plot.sort_values("Wasserstein", ascending=False)
        norm_wass = Normalize(vmin=df_w["Wasserstein"].min(), vmax=df_w["Wasserstein"].max())
        cmap_wass = sns.color_palette("mako", as_cmap=True)
        colors_wass = cmap_wass(norm_wass(df_w["Wasserstein"].values))

        sns.barplot(
            x="Wasserstein", y="Variable",
            data=df_w,
            palette=colors_wass, ax=axes[0]
        )
        axes[0].set_title(f"H{h_dim} — Covariate Importance (Wasserstein)")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("") 

        sm_wass = ScalarMappable(cmap=cmap_wass, norm=norm_wass)
        sm_wass.set_array([])
        fig.colorbar(sm_wass, ax=axes[0], orientation="vertical", label="Importance")

        # ----------- BOTTLENECK ------------
        df_b = df_plot.sort_values("Bottleneck", ascending=False)
        norm_bott = Normalize(vmin=df_b["Bottleneck"].min(), vmax=df_b["Bottleneck"].max())
        cmap_bott = sns.color_palette("rocket", as_cmap=True)
        colors_bott = cmap_bott(norm_bott(df_b["Bottleneck"].values))

        sns.barplot(
            x="Bottleneck", y="Variable",
            data=df_b,
            palette=colors_bott, ax=axes[1]
        )
        axes[1].set_title(f"H{h_dim} — Covariate Importance (Bottleneck)")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("") 

        sm_bott = ScalarMappable(cmap=cmap_bott, norm=norm_bott)
        sm_bott.set_array([])
        fig.colorbar(sm_bott, ax=axes[1], orientation="vertical", label="Importance")

        # Save
        plt.tight_layout()
        out_path = os.path.join(self.args['images'], f"covariate_importance_H{h_dim}.png")
        plt.savefig(out_path, dpi=200)
        print(f"[INFO] Saved H{h_dim} importance plot (with colorbars) to: {out_path}")
        plt.close()
