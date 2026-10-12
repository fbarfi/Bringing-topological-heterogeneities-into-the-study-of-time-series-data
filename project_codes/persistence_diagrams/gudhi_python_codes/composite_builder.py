
from __future__ import annotations
from typing import List, Dict, Tuple

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


DEFAULT_HS: List[int] = [1, 2, 3, 4]  


class FeatureStore:
    """
    Load/normalize/materialize per-H features from the combined '*_H_{Hidx}.csv' files
    produced by GudhiPersistenceRepresentationsClass. Only H∈{1,2,3,4} supported.
    """
    def __init__(self, args):
        self.folder = args["folder"]
        self.ensure_dir(self.folder)
        self.homology_dimensions = args["homology_dimensions"]
        self.Hidx = int("".join(map(str, self.homology_dimensions[1:])))
        
        if self.Hidx != 1234:
            raise ValueError(
                f"Expected homology_dimensions to yield Hidx=1234 (got {self.Hidx}). "
                "Ensure Gudhi writer runs with H={1,2,3,4} only."
            )

    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x)
        lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
        out = (x - lo) / (hi - lo)
        out[~finite] = 0.0
        return out

    @staticmethod
    def _zscore(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True)
        sd[sd == 0.0] = 1.0
        Z = (M - mu) / sd
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z, mu.ravel(), sd.ravel()

    @staticmethod
    def _first_col_like(df: pd.DataFrame, base: str, H: int) -> np.ndarray:
        # exact names produced by your writer
        exacts = [f"{base}_H{H}", f"{base}_H_{H}"]
        for name in exacts:
            if name in df.columns:
                return df[name].to_numpy(dtype=float)
        if base == "H":
            simple = f"H_{H}"
            if simple in df.columns:
                return df[simple].to_numpy(dtype=float)
        # last resort: single numeric column
        num_cols = [c for c in df.columns if c != "year" and np.issubdtype(df[c].dtype, np.number)]
        if len(num_cols) == 1:
            return df[num_cols[0]].to_numpy(dtype=float)
        raise KeyError(f"No column for base='{base}', H={H}. Columns={list(df.columns)}")

    def _read_if_exists(self, path: str) -> pd.DataFrame | None:
        return pd.read_csv(path) if os.path.exists(path) else None

    def _combined_paths(self) -> Dict[str, str]:
        idx = self.Hidx
        return {
            "Amplitude":      os.path.join(self.folder, f"Amplitude_H_{idx}.csv"),
            "Entropy":        os.path.join(self.folder, f"Entropy_H_{idx}.csv"),
            "NumberOfPoints": os.path.join(self.folder, f"NumberOfPoints_H_{idx}.csv"),
            "L2":             os.path.join(self.folder, f"L2_Landscape_H_{idx}.csv"),
            "C1":             os.path.join(self.folder, f"C1_Landscape_H_{idx}.csv"),
            "L1":             os.path.join(self.folder, f"L1_Landscape_H_{idx}.csv"),
        }

    def _assert_h_in_range(self, H: int) -> None:
        if H not in DEFAULT_HS:
            raise ValueError(f"H={H} not supported. Allowed: {DEFAULT_HS} (H=5 removed).")

    def materialize_all_features_for_H(self, H: int) -> str:
        """Build All_Features_Normalized_H{H}.csv from the combined files (H∈{1..4})."""
        self._assert_h_in_range(H)
        files = self._combined_paths()

        series: Dict[str, np.ndarray] = {}
        years: np.ndarray | None = None

        def take_years(d: pd.DataFrame) -> None:
            nonlocal years
            if years is None and "year" in d.columns:
                years = d["year"].to_numpy(dtype=int)

        for key in ["Amplitude", "Entropy", "NumberOfPoints"]:
            d = self._read_if_exists(files[key])
            if d is not None:
                take_years(d)
                series[key] = self._norm01(self._first_col_like(d, base="H", H=H))

        dL2 = self._read_if_exists(files["L2"])
        if dL2 is not None:
            take_years(dL2)
            series["L2"] = self._norm01(self._first_col_like(dL2, base="L2_landscape", H=H))

        dC1 = self._read_if_exists(files["C1"])
        if dC1 is not None:
            take_years(dC1)
            try:
                series["C1"] = self._norm01(self._first_col_like(dC1, base="C1_landscape", H=H))
            except Exception:
                pass

        dL1 = self._read_if_exists(files["L1"])
        if dL1 is not None:
            take_years(dL1)
            try:
                series["L1"] = self._norm01(self._first_col_like(dL1, base="L1_landscape", H=H))
            except Exception:
                pass

        if years is None or not series:
            raise FileNotFoundError(
                f"Cannot assemble features for H{H}. "
                f"Expected combined metric files '*_H_{self.Hidx}.csv' in {self.folder}."
            )

        df = pd.DataFrame({"year": years})
        for k, v in series.items():
            df[k] = v

        out = os.path.join(self.folder, f"All_Features_Normalized_H{H}.csv")
        df.to_csv(out, index=False)
        return out

    def read_overall_for_H(self, H: int) -> pd.DataFrame:
        self._assert_h_in_range(H)
        path = os.path.join(self.folder, f"All_Features_Normalized_H{H}.csv")
        if not os.path.exists(path):
            path = self.materialize_all_features_for_H(H)
        df = pd.read_csv(path)
        if "year" not in df.columns:
            raise ValueError(f"{os.path.basename(path)} must contain 'year'.")
        return df.sort_values("year").reset_index(drop=True)

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> List[str]:
        return [
            c for c in df.columns
            if c != "year" and not c.lower().startswith("composite") and np.issubdtype(df[c].dtype, np.number)
        ]


class CompositePipeline:
    """PCA-based composites + exports/plots (H∈{1,2,3,4})."""
    def __init__(self, args):
        self.folder = args["folder"]
        self.images = args["images"]
        self.store = FeatureStore(args)
        self.store.ensure_dir(self.images)

    @staticmethod
    def _build_pca2(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        years = df["year"].to_numpy(int)
        Z, mu, sd = FeatureStore._zscore(df[feature_cols].to_numpy(float))
        pca = PCA(n_components=2, random_state=0)
        PCs = pca.fit_transform(Z)
        mm1, mm2 = MinMaxScaler(), MinMaxScaler()
        comp1 = mm1.fit_transform(PCs[:, [0]]).ravel()
        comp2 = mm2.fit_transform(PCs[:, [1]]).ravel()
        df_comp = pd.DataFrame({
            "year": years, "pc1": comp1, "pc2": comp2,
            "pc1_raw": PCs[:, 0], "pc2_raw": PCs[:, 1],
        })
        weights = pd.DataFrame({
            "feature": feature_cols,
            "weight_pc1_on_Z": pca.components_[0],
            "weight_pc2_on_Z": pca.components_[1],
        }).sort_values("weight_pc1_on_Z", ascending=False)
        meta = pd.DataFrame({
            "key": [
                "explained_variance_ratio_pc1", "explained_variance_ratio_pc2",
                "mean_by_feature", "std_by_feature",
                "min_pc1_", "scale_pc1_", "min_pc2_", "scale_pc2_",
            ],
            "value": [
                float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1]),
                {f: float(m) for f, m in zip(feature_cols, mu)},
                {f: float(s) for f, s in zip(feature_cols, sd)},
                float(mm1.min_[0]), float(mm1.scale_[0]),
                float(mm2.min_[0]), float(mm2.scale_[0]),
            ],
        })
        return df_comp, weights, meta

    @staticmethod
    def _plot_series(years: np.ndarray, mapping: Dict[str, np.ndarray], title: str, out_png: str) -> None:
        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            plt.figure(figsize=(12, 5))
            for label, arr in mapping.items():
                plt.plot(years, arr, linewidth=1.6, label=label)
            plt.xlabel("Year"); plt.ylabel("Value"); plt.title(title)
            plt.legend(ncol=3); plt.tight_layout()
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=300); plt.close()

    @staticmethod
    def _plot_composite(years: np.ndarray, mapping: Dict[str, np.ndarray], title: str, out_png: str) -> None:
        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            plt.figure(figsize=(12, 5))
            for label, arr in mapping.items():
                plt.plot(years, arr, linewidth=2.0, label=label)
            plt.xlabel("Year"); plt.ylabel("Composite [0,1]"); plt.title(title)
            plt.legend(); plt.tight_layout()
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=300); plt.close()

    @staticmethod
    def _plot_weights_bar(weights_df: pd.DataFrame, col: str, out_png: str, title: str) -> None:
        feats = weights_df["feature"].tolist()
        vals = weights_df[col].to_numpy()
        idx = np.arange(len(feats))
        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            plt.figure(figsize=(10, 5))
            plt.bar(idx, vals)  # keep defaults; avoids style coupling
            plt.xticks(idx, feats, rotation=30, ha="right")
            plt.ylabel("Weight"); plt.title(title)
            plt.tight_layout()
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plt.savefig(out_png, dpi=300); plt.close()

    def _run_for_df(self, df_all: pd.DataFrame, tag: str) -> None:
        feature_cols = self.store.feature_columns(df_all)
        df_comp, weights, meta = self._build_pca2(df_all, feature_cols)

        df_comp.to_csv(os.path.join(self.store.folder, f"Composite_{tag}_unsupervised.csv"), index=False)
        weights.to_csv(os.path.join(self.store.folder, f"Composite_{tag}_unsupervised_weights.csv"), index=False)
        meta.to_csv(os.path.join(self.store.folder, f"Composite_{tag}_unsupervised_meta.csv"), index=False)

        merged_wide = df_all.merge(df_comp, on="year", how="left")
        merged_wide.to_csv(os.path.join(self.store.folder, f"Composite_{tag}_merged_wide.csv"), index=False)

        long_rows = [(int(r["year"]), f, float(r[f])) for _, r in df_all.iterrows() for f in feature_cols]
        df_long = pd.DataFrame(long_rows, columns=["year", "feature", "value"]).merge(df_comp, on="year", how="left")
        df_long.to_csv(os.path.join(self.store.folder, f"Composite_{tag}_merged_long.csv"), index=False)

        self._plot_series(
            years=df_all["year"].to_numpy(),
            mapping={c: df_all[c].to_numpy() for c in feature_cols},
            title=f"Normalized TDA Features ({tag})",
            out_png=os.path.join(self.images, f"All_Features_Normalized_{tag}.png"),
        )
        self._plot_composite(
            years=df_comp["year"].to_numpy(),
            mapping={"Composite (PCA-1)": df_comp["pc1"].to_numpy()},
            title=f"Composite PC1 (Unsupervised) – {tag}",
            out_png=os.path.join(self.images, f"Composite_{tag}_pca1.png"),
        )
        self._plot_composite(
            years=df_comp["year"].to_numpy(),
            mapping={"Composite (PCA-2)": df_comp["pc2"].to_numpy()},
            title=f"Composite PC2 (Unsupervised) – {tag}",
            out_png=os.path.join(self.images, f"Composite_{tag}_pca2.png"),
        )
        self._plot_composite(
            years=df_comp["year"].to_numpy(),
            mapping={
                "Composite (PCA-1)": df_comp["pc1"].to_numpy(),
                "Composite (PCA-2)": df_comp["pc2"].to_numpy(),
            },
            title=f"Composite (PC1 vs PC2) – {tag}",
            out_png=os.path.join(self.images, f"Composite_{tag}_both.png"),
        )
        self._plot_weights_bar(
            weights_df=weights, col="weight_pc1_on_Z",
            out_png=os.path.join(self.images, f"Composite_{tag}_weights_pc1.png"),
            title="PCA-1 Weights (on z-scored features)",
        )
        self._plot_weights_bar(
            weights_df=weights, col="weight_pc2_on_Z",
            out_png=os.path.join(self.images, f"Composite_{tag}_weights_pc2.png"),
            title="PCA-2 Weights (on z-scored features)",
        )

    def build_and_save_composites(
        self,
        homologies: List[int] = DEFAULT_HS,  
        mode: str = "separate",
    ) -> None:
        """
        'separate' = per-H from combined '*_H_{Hidx}.csv' files; 'combined' = across Hs.
        Only H in {1,2,3,4} allowed. Passing H=5 raises ValueError.
        """
        self.store.ensure_dir(self.folder)
        self.store.ensure_dir(self.images)

        bad = sorted(set(homologies) - set(DEFAULT_HS))
        if bad:
            raise ValueError(f"Unsupported homologies requested: {bad}. Allowed: {DEFAULT_HS}")

        if mode == "separate":
            for H in homologies:
                df_all = self.store.read_overall_for_H(H)
                self._run_for_df(df_all, tag=f"H{H}")

        elif mode == "combined":
            prefixed: List[pd.DataFrame] = []
            base_years = None
            for H in homologies:
                dfH = self.store.read_overall_for_H(H)
                years = tuple(dfH["year"].tolist())
                if base_years is None:
                    base_years = years
                elif years != base_years:
                    raise ValueError("Years must match across homologies for combined mode.")
                feat_cols = self.store.feature_columns(dfH)
                dfH = dfH.rename(columns={c: f"H{H}_{c}" for c in feat_cols})
                prefixed.append(dfH)

            df_combined = prefixed[0]
            for k in range(1, len(prefixed)):
                df_combined = df_combined.merge(prefixed[k], on="year", how="inner")

            tag = "H" + "".join(map(str, homologies))
            self._run_for_df(df_combined, tag=tag)
        else:
            raise ValueError("mode must be 'separate' or 'combined'")
