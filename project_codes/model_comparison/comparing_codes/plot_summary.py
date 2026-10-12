from __future__ import annotations
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================== config ===============================

@dataclass(frozen=True)
class PlotConfig:
    nepochs: int = 900
    split: str = "valid"                         # "train" | "valid" | "test"
    cal_start: int = 1945
    cal_end: int = 2010
    target_metrics: Tuple[str, ...] = ("Accuracy", "F1_macro", "AUROC_macro", "WrongPredConfidence") 
    n_bins: int = 15                             # bins for calibration plots

    # filesystem
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    _summary_root: Optional[Path] = None
    _out_dir: Optional[Path] = None

    @property
    def summary_root(self) -> Path:
        if self._summary_root is not None:
            return self._summary_root
        return (self.root / f"../../project_results/tsai_comparing_models_results/nepochs_{self.nepochs}/{self.split}").resolve()

    @property
    def out_dir(self) -> Path:
        if self._out_dir is not None:
            return self._out_dir
        p = (self.summary_root / "summary_plots").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cpd_out(self) -> Path:
        p = (self.out_dir / "CPD").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def calib_out(self) -> Path:
        # Why: keep calibration figures separate and tidy
        p = (self.out_dir / "calibration").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


# ========================== canonicalization ==========================

@dataclass(frozen=True)
class MetricCanonicalizer:
    aliases: Dict[str, str] = field(default_factory=lambda: {
        "accuracy": "Accuracy", "acc": "Accuracy",
        "f1macro": "F1_macro", "f1_macro": "F1_macro",
        "aurocmacro": "AUROC_macro", "auroc_macro": "AUROC_macro",
        "wrongpredconfidence": "WrongPredConfidence",
        "wrong_pred_confidence": "WrongPredConfidence",
        "wrong pred confidence": "WrongPredConfidence",
        "wpc": "WrongPredConfidence",
    })

    @staticmethod
    def _norm_key(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "")

    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if "metric" not in df.columns:
            return df
        out = df.copy()
        keys = out["metric"].astype(str).map(self._norm_key)
        out["metric"] = [self.aliases.get(k, orig.strip()) for k, orig in zip(keys, out["metric"])]
        return out


# ============================ data sources ============================

@dataclass
class SummaryRepository:
    cfg: PlotConfig
    canon: MetricCanonicalizer

    def discover(self) -> List[Path]:
        paths = list(self.cfg.summary_root.glob("*_val_*_test_*_epochs_*/summary.csv"))
        if not paths:
            raise FileNotFoundError(f"No summary.csv under {self.cfg.summary_root}")
        return paths

    def load_all(self) -> Tuple[pd.DataFrame, List[Tuple[Path, pd.DataFrame]]]:
        per_file: List[Tuple[Path, pd.DataFrame]] = []
        bags: List[pd.DataFrame] = []
        for csv in self.discover():
            df = pd.read_csv(csv)
            if "metric" not in df.columns:
                raise KeyError(f"'metric' column missing in {csv}")
            model = csv.parent.name.split("_val_")[0]
            df.insert(0, "model", model)
            df["metric_raw"] = df["metric"].astype(str)
            bags.append(self.canon.canonicalize(df))
            per_file.append((csv, df))
        agg = pd.concat(bags, ignore_index=True)
        agg.to_csv(self.cfg.out_dir / "aggregate_summary.csv", index=False)
        print(f"[info] wrote aggregate: {self.cfg.out_dir/'aggregate_summary.csv'}")
        return agg, per_file


# ===== per-sample repository (for calibration & histograms) ===========

@dataclass
class PerSampleRepository:
    """Loads per-sample predictions emitted by CompareRunner (base vs tda)."""
    def discover(self, summary_csvs: List[Path]) -> List[Path]:
        out = []
        for s in summary_csvs:
            p = s.parent / "per_sample_predictions.csv"
            if p.exists():
                out.append(p)
            else:
                print(f"[warn] missing {p}")
        return out

    def load_all(self, per_sample_csvs: List[Path]) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        for p in per_sample_csvs:
            model = p.parent.name.split("_val_")[0]
            df = pd.read_csv(p)
            need = {"y_true","base_pred","tda_pred","base_conf","tda_conf"}
            if not need.issubset(df.columns):
                print(f"[warn] per-sample missing columns in {p.name}, found {sorted(df.columns)}")
                continue
            d = df.copy()
            d["model"] = model
            d["base_correct"] = (d["base_pred"] == d["y_true"]).astype(int)
            d["tda_correct"]  = (d["tda_pred"]  == d["y_true"]).astype(int)
            keep = ["model","y_true","base_pred","tda_pred","base_conf","tda_conf","base_correct","tda_correct"]
            # carry per-class probability columns (base_p0.., tda_p0..) if present,
            # so macro-OvR reliability/ECE can be computed downstream.
            pcols = [c for c in d.columns if c.startswith("base_p") or c.startswith("tda_p")]
            keep = keep + [c for c in pcols if c not in keep]
            rows.append(d[keep])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[
            "model","y_true","base_pred","tda_pred","base_conf","tda_conf","base_correct","tda_correct"
        ])


# ============================== reporters =============================

@dataclass
class MetricReporter:
    cfg: PlotConfig
    target_metrics: Tuple[str, ...]
    canon: MetricCanonicalizer

    def guard_presence(self, agg_df: pd.DataFrame, per_file_frames: List[Tuple[Path, pd.DataFrame]]) -> None:
        present = sorted(agg_df["metric"].astype(str).unique())
        print("[info] metrics present:", present)
        missing = [m for m in self.target_metrics if m not in present]
        if not missing:
            return

        def _closest(names: List[str], target: str, n: int = 5) -> List[str]:
            return difflib.get_close_matches(target, names, n=n, cutoff=0.55)

        print("\n[error] required target metrics are missing:")
        for miss in missing:
            print(f"  ✖ {miss}")
            aliases = [k for k, v in self.canon.aliases.items() if v == miss]
            if aliases:
                print(f"     accepted aliases (normalized): {aliases}")
            close = _closest(present, miss)
            if close:
                print(f"     close canonical matches seen: {close}")
            print("     per-file raw metric hints:")
            for csv, df in per_file_frames:
                raw = sorted(set(df["metric_raw"].astype(str)))
                norm_map = {MetricCanonicalizer._norm_key(r): r for r in raw}
                candidates = set()
                probe = {MetricCanonicalizer._norm_key(miss)} | {MetricCanonicalizer._norm_key(a) for a in aliases}
                for rk, orig in norm_map.items():
                    if any(difflib.SequenceMatcher(None, rk, pk).ratio() >= 0.7 for pk in probe):
                        candidates.add(orig)
                candidates |= set(_closest(raw, miss, n=3))
                if candidates:
                    print(f"       - {csv.parent.name}/summary.csv → {sorted(candidates)}")
        raise SystemExit(f"Aborting: missing metrics {missing}")

    def _require_cols(self, df: pd.DataFrame, cols: List[str]) -> bool:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[warn] missing columns {missing}; skipping related plots.")
            return False
        return True

    def plot_all(self, agg_df: pd.DataFrame) -> None:
        # ===== Δ (existing) =====
        if self._require_cols(agg_df, ["delta"]):
            avg = agg_df.groupby("metric", as_index=False)["delta"].mean().sort_values("delta", ascending=False)
            plt.figure(figsize=(10, 5))
            sns.barplot(data=avg, x="delta", y="metric", hue="metric", palette="coolwarm", orient="h", legend=False)
            plt.axvline(0, color="gray", linestyle="--")
            plt.title("Average Δ (TDA - Base) Across Models")
            plt.xlabel("Delta"); plt.ylabel("Metric")
            plt.tight_layout(); plt.savefig(self.cfg.out_dir / "avg_deltas_barplot.png", dpi=150); plt.close()

            pivot = agg_df.pivot_table(index="model", columns="metric", values="delta", aggfunc="median")
            order = [m for m in self.target_metrics if m in pivot.columns] + [c for c in pivot.columns if c not in self.target_metrics]
            pivot = pivot.reindex(columns=order)
            plt.figure(figsize=(12, max(4, len(pivot) * 0.5)))
            sns.heatmap(pivot, annot=True, fmt=".3f", center=0, cmap="RdBu_r", linewidths=0.5)
            plt.title("TDA - Base Δ Heatmap")
            plt.tight_layout(); plt.savefig(self.cfg.out_dir / "delta_heatmap.png", dpi=150); plt.close()

            for metric in self.target_metrics:
                df_m = agg_df.loc[agg_df.metric == metric].copy().sort_values("delta", ascending=False)
                if df_m.empty: 
                    continue
                plt.figure(figsize=(10, 5))
                sns.barplot(data=df_m, x="delta", y="model", palette="viridis", orient="h", hue="metric", legend=False)
                plt.axvline(0, color="gray", linestyle="--")
                plt.title(f"Δ (TDA - Base) for {metric}")
                plt.xlabel("Delta"); plt.ylabel("Model")
                plt.tight_layout(); plt.savefig(self.cfg.out_dir / f"delta_{metric}.png", dpi=150); plt.close()

        # ===== Base vs TDA grouped bars  =====
        if self._require_cols(agg_df, ["base", "tda"]):
            mean_bt = (agg_df.groupby("metric")[["base", "tda"]]
                              .mean()
                              .reset_index()
                              .sort_values("tda", ascending=False))
            mean_bt_melt = mean_bt.melt(id_vars="metric", value_vars=["base", "tda"], var_name="model_type", value_name="score")
            plt.figure(figsize=(12, 5))
            sns.barplot(data=mean_bt_melt, x="score", y="metric", hue="model_type", orient="h")
            plt.title("Mean score per metric: Base vs TDA")
            plt.xlabel("Score"); plt.ylabel("Metric"); plt.legend(title="")
            plt.tight_layout(); plt.savefig(self.cfg.out_dir / "mean_base_vs_tda_by_metric.png", dpi=150); plt.close()

            # Paired improvements per metric: base→tda segments per backbone
            for metric in self.target_metrics:
                df_m = agg_df.loc[agg_df.metric == metric, ["model", "base", "tda"]].copy()
                if df_m.empty:
                    continue
                df_m = df_m.sort_values("tda", ascending=False)
                plt.figure(figsize=(12, max(4, len(df_m) * 0.4)))
                y = np.arange(len(df_m))
                plt.scatter(df_m["base"], y, label="base", zorder=3)
                plt.scatter(df_m["tda"], y, label="tda", zorder=3)
                for i, (_, r) in enumerate(df_m.iterrows()):
                    # why: show improvement direction
                    plt.plot([r["base"], r["tda"]], [i, i], alpha=0.5)
                plt.yticks(y, df_m["model"])
                plt.title(f"Base→TDA per model: {metric}")
                plt.xlabel(metric); plt.ylabel("Model"); plt.legend()
                plt.tight_layout(); plt.savefig(self.cfg.out_dir / f"paired_base_to_tda_{metric}.png", dpi=150); plt.close()

        # ===== Wins stacked bars (existing logic; robust) =====
        if "higher_is_better" not in agg_df.columns:
            print("[warn] 'higher_is_better' missing; skipping wins plot.")
        else:
            df_wins = agg_df.copy()
            # direct comparison is equivalent to sign of delta
            if "delta" in df_wins.columns:
                df_wins["tda_wins"] = (
                    (df_wins["higher_is_better"] & (df_wins["delta"] > 0))
                    | (~df_wins["higher_is_better"] & (df_wins["delta"] < 0))
                )
            elif {"base", "tda"}.issubset(df_wins.columns):
                df_wins["tda_wins"] = (
                    (df_wins["higher_is_better"] & (df_wins["tda"] > df_wins["base"]))
                    | (~df_wins["higher_is_better"] & (df_wins["tda"] < df_wins["base"]))
                )
            else:
                print("[warn] neither delta nor base/tda present; skipping wins plot.")
                return
            tda_wins = df_wins.groupby("model")["tda_wins"].sum()
            total = df_wins.groupby("model")["metric"].count()
            base_wins = total - tda_wins
            order = tda_wins.sort_values(ascending=False).index
            tda_wins, base_wins = tda_wins[order], base_wins[order]
            plt.figure(figsize=(12, 5))
            plt.bar(order, base_wins, 0.8, label="Base wins")
            plt.bar(order, tda_wins, 0.8, bottom=base_wins, label="TDA wins")
            plt.title("Base vs TDA Wins per Backbone"); plt.ylabel("Number of metric wins")
            plt.xticks(rotation=45, ha="right"); plt.legend()
            plt.tight_layout(); plt.savefig(self.cfg.out_dir / "base_tda_wins_per_backbone.png", dpi=150, bbox_inches="tight"); plt.close()


# ========================== calibration reporter ======================

@dataclass
class ReliabilityReporter:
    cfg: PlotConfig

    @staticmethod
    def _bin_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
        """Bin confidences → per-bin mean confidence, accuracy, count, ECE contrib."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(conf, bins) - 1
        rows = []
        for b in range(n_bins):
            m = idx == b
            if not np.any(m):
                continue
            c_mean = float(np.mean(conf[m]))
            acc = float(np.mean(correct[m]))
            frac = float(np.mean(m))
            rows.append(dict(bin=b, conf_mean=c_mean, acc=acc, frac=frac, gap=abs(acc - c_mean)))
        return pd.DataFrame(rows)

    @staticmethod
    def _ece(df_bins: pd.DataFrame) -> float:
        return float(np.sum(df_bins["frac"] * df_bins["gap"])) if not df_bins.empty else float("nan")

    @staticmethod
    def _ece_binary_ovr(pos: np.ndarray, p1: np.ndarray, n_bins: int) -> float:
        """One-vs-rest ECE for a single class: bin by predicted prob p1, compare
        mean predicted prob (confidence) to empirical positive fraction in each
        bin. IDENTICAL math to ece_binary in comparing_tsai_models.py so the
        figure and the calibration table report the SAME macro-OvR ECE."""
        pos = np.asarray(pos, dtype=float).reshape(-1)
        p1 = np.asarray(p1, dtype=float).reshape(-1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p1, bins) - 1, 0, n_bins - 1)
        ece = 0.0; n = len(p1)
        for b in range(n_bins):
            m = idx == b; cnt = int(m.sum())
            if cnt == 0:
                continue
            ece += (cnt / n) * abs(pos[m].mean() - p1[m].mean())
        return float(ece)

    def _macro_ovr_ece(self, df: pd.DataFrame, who: str) -> Optional[float]:
        """Macro-OvR ECE for model `who` in {'base','tda'} using per-class
        columns who_p0.. if present; returns None if they are absent."""
        # Match ONLY per-class probability columns "<who>_p<digits>" (e.g.
        # base_p0, tda_p3) -- NOT base_pred / tda_pred, which also start with
        # "<who>_p". Sort by the trailing class index.
        _pat = re.compile(rf"^{re.escape(who)}_p(\d+)$")
        _matches = [(int(m.group(1)), c) for c in df.columns
                    for m in [_pat.match(c)] if m]
        pcols = [c for _, c in sorted(_matches, key=lambda t: t[0])]
        if not pcols:
            return None
        y = df["y_true"].to_numpy()
        eces = []
        for c, col in enumerate(pcols):
            pos = (y == c).astype(int)
            eces.append(self._ece_binary_ovr(pos, df[col].to_numpy(), self.cfg.n_bins))
        return float(np.mean(eces)) if eces else None

    def _ovr_reliability_curve(self, df: pd.DataFrame, who: str) -> pd.DataFrame:
        """Per-class-pooled reliability points for plotting: pool all (p_c, 1{y=c})
        pairs across classes, then bin. Gives one averaged OvR reliability curve."""
        # Match ONLY per-class probability columns "<who>_p<digits>" (e.g.
        # base_p0, tda_p3) -- NOT base_pred / tda_pred, which also start with
        # "<who>_p". Sort by the trailing class index.
        _pat = re.compile(rf"^{re.escape(who)}_p(\d+)$")
        _matches = [(int(m.group(1)), c) for c in df.columns
                    for m in [_pat.match(c)] if m]
        pcols = [c for _, c in sorted(_matches, key=lambda t: t[0])]
        y = df["y_true"].to_numpy()
        confs, poss = [], []
        for c, col in enumerate(pcols):
            confs.append(df[col].to_numpy())
            poss.append((y == c).astype(int))
        conf = np.concatenate(confs); pos = np.concatenate(poss).astype(int)
        return self._bin_stats(conf, pos, self.cfg.n_bins)

    def plot_reliability_pair(self, df: pd.DataFrame, title: str, out_path: Path) -> None:
        # Prefer macro-OvR (per-class) reliability if per-class probs are present;
        # fall back to top-label reliability otherwise (old CSVs).
        ece_b_ovr = self._macro_ovr_ece(df, "base")
        ece_t_ovr = self._macro_ovr_ece(df, "tda")
        use_ovr = (ece_b_ovr is not None) and (ece_t_ovr is not None)

        if use_ovr:
            b = self._ovr_reliability_curve(df, "base")
            t = self._ovr_reliability_curve(df, "tda")
            ece_b, ece_t = ece_b_ovr, ece_t_ovr
            xlab, ylab = "Predicted probability (OvR)", "Empirical frequency"
            ece_label = "macro-OvR ECE"
        else:
            b = self._bin_stats(df["base_conf"].to_numpy(), df["base_correct"].to_numpy(), self.cfg.n_bins)
            t = self._bin_stats(df["tda_conf"].to_numpy(),  df["tda_correct"].to_numpy(),  self.cfg.n_bins)
            ece_b, ece_t = self._ece(b), self._ece(t)
            xlab, ylab = "Confidence", "Accuracy"
            ece_label = "ECE"

        plt.figure(figsize=(6.5, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.plot(b["conf_mean"], b["acc"], marker="o", label=f"base ({ece_label}={ece_b:.3f})")
        plt.plot(t["conf_mean"], t["acc"], marker="o", label=f"tda ({ece_label}={ece_t:.3f})")
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

    def plot_conf_hist_pair(self, df: pd.DataFrame, title: str, out_path: Path) -> None:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(df["base_conf"], label="base", fill=True, alpha=0.3, bw_adjust=0.8)
        sns.kdeplot(df["tda_conf"],  label="tda",  fill=True, alpha=0.3, bw_adjust=0.8)
        plt.xlim(0, 1); plt.xlabel("Confidence"); plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

    def _family_binned(self, per_sample_df: pd.DataFrame, who: str):
        """Evaluate each backbone's macro-OvR reliability on a COMMON fixed bin
        grid so the per-backbone accuracies are stackable per bin. Returns
        (centers[n_bins], acc_matrix[n_models, n_bins] with NaNs where a model
        has no mass in a bin, model_names, per_model_ece, pooled_center,
        pooled_acc, pooled_ece)."""
        n_bins = self.cfg.n_bins
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        def binned_acc(df):
            # pool per-class (p_c, 1{y=c}) then bin on the fixed grid
            _pat = re.compile(rf"^{re.escape(who)}_p(\d+)$")
            matches = [(int(m.group(1)), c) for c in df.columns
                       for m in [_pat.match(c)] if m]
            pcols = [c for _, c in sorted(matches, key=lambda t: t[0])]
            if not pcols:
                return None
            y = df["y_true"].to_numpy()
            confs, poss = [], []
            for c, col in enumerate(pcols):
                confs.append(df[col].to_numpy())
                poss.append((y == c).astype(int))
            conf = np.concatenate(confs); pos = np.concatenate(poss).astype(int)
            idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
            acc = np.full(n_bins, np.nan)
            for b in range(n_bins):
                m = idx == b
                if m.any():
                    acc[b] = pos[m].mean()
            return acc

        names, rows, eces = [], [], []
        for model, g in sorted(per_sample_df.groupby("model"),
                               key=lambda kv: str(kv[0]).lower()):
            a = binned_acc(g)
            if a is None:
                continue
            names.append(str(model)); rows.append(a)
            eces.append(self._macro_ovr_ece(g, who))
        acc_mat = np.vstack(rows) if rows else np.empty((0, n_bins))
        pooled_acc = binned_acc(per_sample_df)
        pooled_ece = self._macro_ovr_ece(per_sample_df, who)
        return centers, acc_mat, names, eces, pooled_acc, pooled_ece

    def plot_reliability_family(self, per_sample_df: pd.DataFrame, who: str,
                                title: str, out_path: Path,
                                style: str = "spread") -> None:
        """Cross-backbone macro-OvR reliability for one family (`who` in
        {'base','tda'}), plotted so the spread across backbones is legible.

        style='spread' (default): a single pooled curve with a vertical bar at
            each bin spanning the min-max of the backbones' empirical frequency
            in that bin, plus a lighter +/-1 SD band; faint per-backbone dots
            underneath. This is the calibration analogue of the grouped
            accuracy plot with error bars.
        style='markers': every backbone as unconnected dots (no lines), plus the
            pooled curve. Less crowded than connected polylines but keeps each
            backbone identifiable by color.

        Reuses the same per-class binning as the calibration table, so the
        pooled ECE in the legend matches it. Skips if per-class probs absent."""
        if self._macro_ovr_ece(per_sample_df, who) is None:
            print(f"[warn] no per-class {who} probs; skipping {out_path.name}")
            return

        centers, acc_mat, names, eces, pooled_acc, pooled_ece = \
            self._family_binned(per_sample_df, who)

        plt.figure(figsize=(6.5, 6.5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                 label="perfect calibration")

        if style == "markers":
            cmap = plt.get_cmap("tab10")
            for i, name in enumerate(names):
                a = acc_mat[i]
                ok = ~np.isnan(a)
                plt.plot(centers[ok], a[ok], marker="o", ms=4, lw=0,
                         color=cmap(i % 10), alpha=0.6,
                         label=f"{name} ({eces[i]:.3f})")
        else:  # spread
            # faint per-backbone dots for context
            for i in range(acc_mat.shape[0]):
                a = acc_mat[i]; ok = ~np.isnan(a)
                plt.plot(centers[ok], a[ok], marker="o", ms=2.5, lw=0,
                         color="#4C72B0", alpha=0.18, zorder=1)
            # min-max whiskers and +/-1 SD band across backbones per bin
            with np.errstate(invalid="ignore"):
                lo = np.nanmin(acc_mat, axis=0)
                hi = np.nanmax(acc_mat, axis=0)
                mean = np.nanmean(acc_mat, axis=0)
                sd = np.nanstd(acc_mat, axis=0)
            ok = ~np.isnan(mean)
            plt.fill_between(centers[ok], (mean - sd)[ok], (mean + sd)[ok],
                             color="#4C72B0", alpha=0.20, zorder=1,
                             label=r"$\pm1$ SD across backbones")
            plt.vlines(centers[ok], lo[ok], hi[ok], color="#4C72B0",
                       alpha=0.55, lw=1.2, zorder=2,
                       label="min-max across backbones")

        # pooled curve on top
        okp = ~np.isnan(pooled_acc)
        plt.plot(centers[okp], pooled_acc[okp], marker="s", ms=5, lw=2.6,
                 color="black", zorder=3, label=f"pooled ({pooled_ece:.3f})")

        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("Predicted probability (OvR)")
        plt.ylabel("Empirical frequency")
        plt.title(title)
        plt.legend(loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()

    def plot_reliability_family_compare(self, per_sample_df: pd.DataFrame,
                                        title: str, out_path: Path) -> None:
        """Overlay BOTH families' across-backbone calibration on one axes in the
        spread style: for each of base and TDA, a pooled macro-OvR reliability
        curve with a per-bin min-max whisker and a +/-1 SD band across the ten
        backbones. Colors are chosen to discriminate the two families (TDA blue,
        base red), matching the TDA/base convention used elsewhere. Skips a
        family if its per-class probability columns are absent."""
        families = [
            ("tda",  "TDA",  "#4C72B0"),   # blue
            ("base", "base", "#C44E52"),   # red
        ]
        drew = False
        plt.figure(figsize=(6.8, 6.8))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                 label="perfect calibration", zorder=1)
        for who, lab, col in families:
            if self._macro_ovr_ece(per_sample_df, who) is None:
                print(f"[warn] no per-class {who} probs; omitting {lab} from "
                      f"{out_path.name}")
                continue
            centers, acc_mat, names, eces, pooled_acc, pooled_ece = \
                self._family_binned(per_sample_df, who)
            with np.errstate(invalid="ignore"):
                lo = np.nanmin(acc_mat, axis=0)
                hi = np.nanmax(acc_mat, axis=0)
                mean = np.nanmean(acc_mat, axis=0)
                sd = np.nanstd(acc_mat, axis=0)
            okb = ~np.isnan(mean)
            plt.fill_between(centers[okb], (mean - sd)[okb], (mean + sd)[okb],
                             color=col, alpha=0.18, zorder=1)
            plt.vlines(centers[okb], lo[okb], hi[okb], color=col, alpha=0.45,
                       lw=1.1, zorder=2)
            okp = ~np.isnan(pooled_acc)
            plt.plot(centers[okp], pooled_acc[okp], marker="o", ms=4.5, lw=2.4,
                     color=col, zorder=3,
                     label=f"{lab} pooled (ECE {pooled_ece:.3f}); "
                           f"band $\\pm1$ SD, whisker min-max")
            drew = True
        if not drew:
            plt.close()
            print(f"[warn] neither family had per-class probs; skipping {out_path.name}")
            return
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.gca().set_aspect("equal", "box")
        plt.xlabel("Predicted probability (OvR)")
        plt.ylabel("Empirical frequency")
        plt.title(title)
        plt.legend(loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()

    def plot_all(self, per_sample_df: pd.DataFrame) -> None:
        if per_sample_df.empty:
            print("[warn] no per-sample predictions available; skipping calibration plots.")
            return

        # Per-model
        for model, g in per_sample_df.groupby("model"):
            title = f"Reliability: {model} (base vs tda)"
            out_rel = self.cfg.calib_out / f"reliability_{model}.png"
            self.plot_reliability_pair(g, title, out_rel)

            title_h = f"Confidence Histogram: {model} (base vs tda)"
            out_hist = self.cfg.calib_out / f"conf_hist_{model}.png"
            self.plot_conf_hist_pair(g, title_h, out_hist)

        # Aggregate (base-vs-tda pooled over all backbones)
        self.plot_reliability_pair(
            per_sample_df, "Reliability: ALL models (base vs tda)", self.cfg.calib_out / "reliability_ALL.png"
        )
        self.plot_conf_hist_pair(
            per_sample_df, "Confidence Histogram: ALL models (base vs tda)", self.cfg.calib_out / "conf_hist_ALL.png"
        )

        # Per-family overlays: every backbone summarized on one axes, TDA and
        # base separately. Two styles emitted for each family so you can choose:
        # '*_spread.png' (pooled curve + across-backbone min-max/SD spread) and
        # '*_markers.png' (unconnected per-backbone dots). Pick one for the paper.
        for who, tag in (("tda", "tda"), ("base", "base")):
            title = f"Macro-OvR reliability - {who.upper()} backbones"
            self.plot_reliability_family(
                per_sample_df, who, title,
                self.cfg.calib_out / f"reliability_{tag}_all_backbones_spread.png",
                style="spread",
            )
            self.plot_reliability_family(
                per_sample_df, who, title,
                self.cfg.calib_out / f"reliability_{tag}_all_backbones_markers.png",
                style="markers",
            )

        # Both families on one axes (spread style), TDA vs base, for direct
        # side-by-side comparison of across-backbone calibration.
        self.plot_reliability_family_compare(
            per_sample_df,
            "Macro-OvR reliability across backbones - TDA vs base",
            self.cfg.calib_out / "reliability_tda_vs_base_all_backbones_spread.png",
        )


# ================================ CPD =================================

@dataclass(frozen=True)
class CPDFileMap:
    # optional data-only curves
    tda_signal: str = "cpd_tda_signal.csv"
    base_signal: str = "cpd_base_signal.csv"
    # delta curves (provided by compare step)
    delta_nll:  str = "cpd_delta_model_nll.csv"
    delta_margin: str = "cpd_delta_model_margin.csv"
    # raw per-model time curves (for overlays)
    base_nll_time: str = "cpd_base_model_nll_time.csv"
    tda_nll_time:  str = "cpd_tda_model_nll_time.csv"
    base_margin_time: str = "cpd_base_model_margin_time.csv"
    tda_margin_time:  str = "cpd_tda_model_margin_time.csv"


@dataclass
class CPDRepository:
    cfg: PlotConfig
    files: CPDFileMap = field(default_factory=CPDFileMap)

    def _read_one(self, model_dir: Path, filename: str) -> Optional[pd.DataFrame]:
        p = model_dir / filename
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if not {"year", "score"}.issubset(df.columns):
            return None
        df = df[["year", "score"]].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["year", "score"])
        df = df[(df["year"] >= self.cfg.cal_start) & (df["year"] <= self.cfg.cal_end)]
        if df.empty:
            return None
        df = df.groupby("year", as_index=False)["score"].mean().sort_values("year")
        ys = df["year"].to_numpy()
        if not np.all(np.diff(ys) > 0):
            return None
        return df.reset_index(drop=True)

    def collect(self, summary_csvs: List[Path], kind: str) -> pd.DataFrame:
        filename = getattr(self.files, kind)
        rows: List[pd.DataFrame] = []
        for csv in summary_csvs:
            model_dir = csv.parent
            model = model_dir.name.split("_val_")[0]
            df = self._read_one(model_dir, filename)
            if df is None:
                continue
            d = df.copy(); d["model"] = model; d["kind"] = kind
            rows.append(d[["model", "year", "kind", "score"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model", "year", "kind", "score"])

    def collect_pair(self, summary_csvs: List[Path], base_file: str, tda_file: str, out_kind: str) -> pd.DataFrame:
        # Why: overlay base vs tda for same kind
        rows: List[pd.DataFrame] = []
        for csv in summary_csvs:
            model_dir = csv.parent
            model = model_dir.name.split("_val_")[0]
            b = self._read_one(model_dir, base_file)
            t = self._read_one(model_dir, tda_file)
            if b is None and t is None:
                continue
            if b is not None:
                db = b.copy(); db["model"] = model; db["kind"] = out_kind; db["who"] = "base"
                rows.append(db[["model", "year", "kind", "who", "score"]])
            if t is not None:
                dt = t.copy(); dt["model"] = model; dt["kind"] = out_kind; dt["who"] = "tda"
                rows.append(dt[["model", "year", "kind", "who", "score"]])
        cols = ["model", "year", "kind", "who", "score"]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


@dataclass
class CPDReporter:
    cfg: PlotConfig

    def plot_all_models(self, df_long: pd.DataFrame, kind: str, out_path: Path, smooth_window: Optional[int] = 3) -> None:
        d = df_long[df_long["kind"] == kind].copy()
        if d.empty:
            print(f"[warn] no CPD data for kind='{kind}'")
            return
        ycol = "score"
        if smooth_window and smooth_window > 1:
            d["score_sm"] = d.groupby("model", group_keys=False)["score"].transform(
                lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
            )
            ycol = "score_sm"
        plt.figure(figsize=(12, 6))
        for model, g in d.groupby("model"):
            plt.plot(g["year"], g[ycol], label=model, linewidth=2, alpha=0.9)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.title(f"CPD (time) across backbones – {kind}")
        plt.xlabel("Year"); plt.ylabel("Score")
        plt.legend(title="Model", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[done] saved CPD plot → {out_path}")

    def plot_overlay(self, df_long: pd.DataFrame, kind: str, out_path: Path, smooth_window: Optional[int] = 3) -> None:
        d = df_long[df_long["kind"] == kind].copy()
        if d.empty:
            print(f"[warn] no overlay CPD data for kind='{kind}'")
            return
        ycol = "score"
        if smooth_window and smooth_window > 1:
            d["score_sm"] = d.groupby(["model", "who"], group_keys=False)["score"].transform(
                lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean()
            )
            ycol = "score_sm"
        plt.figure(figsize=(12, 6))
        for (model, who), g in d.groupby(["model", "who"]):
            label = f"{model} – {who}"
            plt.plot(g["year"], g[ycol], label=label, linewidth=2, alpha=0.9)
        if "delta" in kind or "nll" in kind or "margin" in kind:
            plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.title(f"CPD overlay (base vs tda) – {kind}")
        plt.xlabel("Year"); plt.ylabel("Score")
        plt.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[done] saved overlay CPD plot → {out_path}")


# ================================ runner ================================

@dataclass
class ReportRunner:
    cfg: PlotConfig
    canon: MetricCanonicalizer = field(default_factory=MetricCanonicalizer)

    def run(self) -> None:
        repo = SummaryRepository(self.cfg, self.canon)
        agg_df, per_file_frames = repo.load_all()

        metric_reporter = MetricReporter(self.cfg, self.cfg.target_metrics, self.canon)
        metric_reporter.guard_presence(agg_df, per_file_frames)
        metric_reporter.plot_all(agg_df)

        cpd_repo = CPDRepository(self.cfg)
        cpd_reporter = CPDReporter(self.cfg)

        summary_csvs = [p for p, _ in per_file_frames]

        # ===== Existing Δ plots =====
        for kind in ("delta_nll", "delta_margin", "tda_signal"):
            all_cpd = cpd_repo.collect(summary_csvs, kind)
            if not all_cpd.empty:
                cpd_reporter.plot_all_models(all_cpd, kind, self.cfg.out_dir / f"all_models_cpd_{kind}.png", smooth_window=3)

        # ===== NEW: overlay comparisons base vs tda =====
        sig_overlay = cpd_repo.collect_pair(summary_csvs, cpd_repo.files.base_signal, cpd_repo.files.tda_signal, out_kind="signal")
        if not sig_overlay.empty:
            cpd_reporter.plot_overlay(sig_overlay, "signal", self.cfg.cpd_out / "overlay_signal_base_vs_tda.png", smooth_window=3)

        nll_overlay = cpd_repo.collect_pair(summary_csvs, cpd_repo.files.base_nll_time, cpd_repo.files.tda_nll_time, out_kind="nll_time")
        if not nll_overlay.empty:
            cpd_reporter.plot_overlay(nll_overlay, "nll_time", self.cfg.cpd_out / "overlay_nll_time_base_vs_tda.png", smooth_window=3)

        mar_overlay = cpd_repo.collect_pair(summary_csvs, cpd_repo.files.base_margin_time, cpd_repo.files.tda_margin_time, out_kind="margin_time")
        if not mar_overlay.empty:
            cpd_reporter.plot_overlay(mar_overlay, "margin_time", self.cfg.cpd_out / "overlay_margin_time_base_vs_tda.png", smooth_window=3)

        # ===== NEW: calibration diagnostics (reliability + histograms) =====
        ps_repo = PerSampleRepository()
        ps_files = ps_repo.discover(summary_csvs)
        ps_df = ps_repo.load_all(ps_files)
        ReliabilityReporter(self.cfg).plot_all(ps_df)

