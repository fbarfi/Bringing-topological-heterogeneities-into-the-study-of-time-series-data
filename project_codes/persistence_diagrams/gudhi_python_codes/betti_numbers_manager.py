# betti_manager.py

import os
import glob
import shutil
import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0; both compute the same rule.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Iterable, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler


class BettiNumbersManager:
    """
    Standalone manager for Betti curves and summaries.
    Works with PD CSVs produced as: PersistenceDiagrams/PD_H_{Hidx}_{i}.csv
    Produces outputs under: BettiNumbers/
    """

    def __init__(self, args):
        self.folder = args["folder"]
        self.homology_dimensions = args["homology_dimensions"]
        self.n_years = args["n_years"]
        self.Hidx = int("".join(map(str, self.homology_dimensions[1:])))
        self.pd_dir = os.path.join(self.folder, "PersistenceDiagrams")
        self.betti_dir = os.path.join(self.folder, "BettiNumbers")
        os.makedirs(self.betti_dir, exist_ok=True)
        self._pd_pattern = os.path.join(self.pd_dir, f"PD_H_{self.Hidx}_*.csv")

    def _find_pd_files(self) -> List[str]:
        files = sorted(glob.glob(self._pd_pattern))
        if not files:
            raise FileNotFoundError(f"No PD files found: {self._pd_pattern}")
        return files

    def _load_pds_by_dim(self) -> Dict[int, List[np.ndarray]]:
        files = self._find_pd_files()
        dims = self.homology_dimensions[1:]
        by_dim = {k: [] for k in dims}
        for fp in files:
            df = pd.read_csv(fp)
            need = {"b", "d", "h_dim"}
            if not need.issubset(df.columns):
                raise ValueError(f"{fp} must contain columns {need}")
            grouped = dict(tuple(df.groupby("h_dim")))
            for k in dims:
                if k in grouped:
                    arr = grouped[k][["b", "d"]].to_numpy(dtype=float)
                else:
                    arr = np.zeros((0, 2), dtype=float)
                by_dim[k].append(arr)
        return by_dim

    @staticmethod
    def _betti_curve(diag: np.ndarray, grid: np.ndarray) -> np.ndarray:
        if diag.size == 0:
            return np.zeros_like(grid)
        b = diag[:, 0].astype(float)
        d = diag[:, 1].astype(float)
        m = np.isfinite(b) & np.isfinite(d) & (d > b)
        if not np.any(m):
            return np.zeros_like(grid)
        b = b[m].reshape(-1, 1)
        d = d[m].reshape(-1, 1)
        trow = grid[None, :]
        alive = (b <= trow) & (trow < d)
        return alive.sum(axis=0).astype(int)

    @staticmethod
    def _safe_span(t: np.ndarray) -> float:
        if t.size < 2:
            return 0.0
        a, b = t[0], t[-1]
        if not (np.isfinite(a) and np.isfinite(b)):
            return 0.0
        return float(b - a)

    def _list_betti_entries(self) -> List[Tuple[int, int, str]]:
        pattern = os.path.join(self.betti_dir, "csv_files", f"betti_{self.Hidx}_*.csv")
        files = sorted(glob.glob(pattern))
        entries: List[Tuple[int, int, str]] = []
        for fp in files:
            df = pd.read_csv(fp)
            if not {"index", "year", "t"}.issubset(df.columns):
                continue
            entries.append((int(df["index"].iloc[0]), int(df["year"].iloc[0]), fp))
        entries.sort(key=lambda x: x[0])
        return entries

    def global_scale_pds(self, in_place: bool = True, backup: bool = True) -> Dict[str, Any]:
        files = self._find_pd_files()
        stacks = []
        for fp in files:
            df = pd.read_csv(fp)
            if not {"b", "d"}.issubset(df.columns):
                continue
            arr = df[["b", "d"]].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if arr.size:
                stacks.append(arr)
        if not stacks:
            raise ValueError("No finite (b,d) values to fit scaling.")
        X = np.vstack(stacks)
        scaler = MinMaxScaler().fit(X)
        out_dir = self.pd_dir

        for fp in files:
            df = pd.read_csv(fp)
            arr = df[["b", "d"]].to_numpy(dtype=float)
            finite = np.isfinite(arr).all(axis=1)
            arr_scaled = arr.copy()
            if finite.any():
                arr_scaled[finite] = scaler.transform(arr[finite])
            df.loc[:, "b"] = arr_scaled[:, 0]
            df.loc[:, "d"] = arr_scaled[:, 1]
            out_fp = os.path.join(out_dir, os.path.basename(fp))
            df.to_csv(out_fp, index=False)

        return {
            "files": len(files),
            "out_dir": out_dir,
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
        }

    def betti_numbers(
        self,
        resolution: int = 200,
        normalize_t: bool = True,
    ) -> Dict[str, Any]:
        dims = self.homology_dimensions[1:]
        pd_by_dim = self._load_pds_by_dim()

        all_b, all_d = [], []
        for k in dims:
            for diag in pd_by_dim[k]:
                if diag.size:
                    b, d = diag[:, 0], diag[:, 1]
                    m = np.isfinite(b) & np.isfinite(d) & (d > b)
                    if np.any(m):
                        all_b.append(b[m])
                        all_d.append(d[m])
        if all_b and all_d:
            min_b = float(np.min([np.min(x) for x in all_b]))
            max_d = float(np.max([np.max(x) for x in all_d]))
        else:
            min_b, max_d = 0.0, 1.0
        if not np.isfinite(min_b):
            min_b = 0.0
        if not np.isfinite(max_d):
            max_d = 1.0
        if max_d <= min_b:
            eps = max(1e-6, abs(max_d - min_b) + 1e-6)
            max_d = min_b + eps

        t = np.linspace(min_b, max_d, int(resolution))
        t_norm = (t - t[0]) / (t[-1] - t[0]) if t.size > 1 else np.zeros_like(t)

        idx_path = os.path.join(self.betti_dir, "index.csv")
        pd.DataFrame({"index": np.arange(len(self.n_years), dtype=int), "year": self.n_years}).to_csv(idx_path, index=False)

        curves_by_dim: Dict[int, List[np.ndarray]] = {k: [] for k in dims}
        for k in dims:
            curves_by_dim[k] = [self._betti_curve(diag, t) for diag in pd_by_dim[k]]

        plots = []
        plots_dir = os.path.join(self.betti_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for k in dims:
            if not curves_by_dim[k]:
                continue
            plt.figure()
            any_plotted = False
            for i, beta in enumerate(curves_by_dim[k]):
                x = t_norm if normalize_t else t
                m = np.asarray(beta) > 0
                if np.any(m):
                    label = str(self.n_years[i]) if i < len(self.n_years) else f"idx {i}"
                    plt.plot(x[m], beta[m], label=label)
                    any_plotted = True
            if not any_plotted:
                plt.close()
                continue
            plt.xlabel("normalized t" if normalize_t else "t")
            plt.ylabel(rf"Betti $\beta_{{{k}}}(t)$")
            plt.title(f"Betti Curves (H{k})")
            plt.tight_layout()
            outp = os.path.join(plots_dir, f"betti_H{k}.png")
            plt.savefig(outp, dpi=200)
            plt.close()
            plots.append(outp)

        written = []
        csv_dir = os.path.join(self.betti_dir, "csv_files")
        dim_cols = [f"H{k}" for k in dims]
        for i, yr in enumerate(self.n_years):
            row = {
                "index": np.full_like(t, i, dtype=int),
                "year": np.full_like(t, yr, dtype=int),
                "t": t.astype(float),
                "t_norm": t_norm.astype(float),
            }
            for k in dims:
                row[f"H{k}"] = curves_by_dim[k][i]
            cols = ["index", "year", "t", "t_norm"] + dim_cols
            df_year = pd.DataFrame(row, columns=cols)
            self.csv_files = os.path.join(self.betti_dir, "csv_files/")
            os.makedirs(csv_dir, exist_ok=True)
            out_fp = os.path.join(csv_dir, f"betti_{self.Hidx}_{i}.csv")
            df_year.to_csv(out_fp, index=False)
            written.append(out_fp)

        return {"index_csv": idx_path, "year_files": written, "plots": plots}

    def betti_summary(
        self,
        metrics: Iterable[str] = ("area", "mean", "max", "l2"),
        use_t_norm: bool = False,
    ) -> Dict[str, Any]:
        dims = self.homology_dimensions[1:]
        entries = self._list_betti_entries()

        def _metric_vals(t: np.ndarray, beta: np.ndarray, which: str) -> float:
            if beta.size == 0 or t.size == 0:
                return 0.0
            if which == "area":
                return float(_trapezoid(beta, t))
            if which == "mean":
                sp = self._safe_span(t)
                return float(_trapezoid(beta, t) / sp) if sp > 0 else float(np.mean(beta))
            if which == "max":
                return float(np.max(beta))
            if which == "l2":
                return float(np.sqrt(_trapezoid(beta * beta, t)))
            raise ValueError("Unknown metric")

        all_rows = []
        plots = []
        plots_dir = os.path.join(self.betti_dir, "summary", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for k in dims:
            xs_year, series = [], {m: [] for m in metrics}
            for idx, yr, fp in entries:
                df = pd.read_csv(fp)
                t = df["t_norm"].to_numpy(float) if use_t_norm and "t_norm" in df.columns else df["t"].to_numpy(float)
                beta = df.get(f"H{k}", pd.Series(np.zeros_like(t))).to_numpy(float)
                row = {"index": idx, "year": yr, "dim": k}
                for m in metrics:
                    row[m] = _metric_vals(t, beta, m)
                    series[m].append(row[m])
                all_rows.append(row)
                xs_year.append(yr)
            plt.figure()
            for m in metrics:
                plt.plot(xs_year, series[m], label=m)
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.title(f"Betti summary metrics (H{k})")
            plt.legend(loc="best")
            plt.tight_layout()
            outp = os.path.join(plots_dir, f"BettiSummary_H{k}.png")
            plt.savefig(outp, dpi=150)
            plt.close()
            plots.append(outp)

        out_csv = os.path.join(self.betti_dir, "summary", "BettiSummary_ALL.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        return {"summary_csv": out_csv, "plots": plots}

    def betti_timeseries(
        self,
        stat: str = "area",
        t_quantile: Optional[float] = None,
    ) -> Dict[str, Any]:
        dims = self.homology_dimensions[1:]
        entries = self._list_betti_entries()

        def _span(t):
            return float(t[-1] - t[0]) if t.size > 1 and np.isfinite([t[0], t[-1]]).all() else 0.0

        rows: List[Dict[str, Any]] = []
        series_by_dim: Dict[int, Tuple[List[int], List[float]]] = {}

        for k in dims:
            xs_year: List[int] = []
            ys_val: List[float] = []
            for idx, yr, fp in entries:
                df = pd.read_csv(fp)
                t = df["t"].to_numpy(float)
                hk = df.get(f"H{k}", pd.Series(np.zeros_like(t))).to_numpy(float)

                if hk.size == 0 or t.size == 0:
                    val = 0.0
                elif stat == "area":
                    val = float(_trapezoid(hk, t))
                elif stat == "mean":
                    sp = _span(t)
                    val = float(_trapezoid(hk, t) / sp) if sp > 0 else float(np.mean(hk))
                elif stat == "max":
                    val = float(np.max(hk))
                elif stat == "l2":
                    val = float(np.sqrt(_trapezoid(hk * hk, t)))
                elif stat == "quantile":
                    if not (t_quantile and 0.0 < t_quantile <= 1.0):
                        raise ValueError("Provide t_quantile in (0,1] for stat='quantile'")
                    if t.size:
                        target = t[0] + t_quantile * (t[-1] - t[0])
                        j = int(np.argmin(np.abs(t - target)))
                        val = float(hk[j])
                    else:
                        val = 0.0
                else:
                    raise ValueError("stat must be one of: area, mean, max, l2, quantile")

                xs_year.append(yr)
                ys_val.append(val)
                rows.append({"index": idx, "year": yr, "dim": k, "stat": stat, "value": val})

            series_by_dim[k] = (xs_year, ys_val)

        plots_dir = os.path.join(self.betti_dir, "summary", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure()
        for k in dims:
            xs, ys = series_by_dim[k]
            plt.plot(xs, ys, label=f"H{k}")
        plt.xlabel("Year")
        plt.ylabel(stat)
        plt.title(f"Betti {stat} over time (all dimensions)")
        plt.legend(loc="best")
        plt.tight_layout()
        out_plot = os.path.join(plots_dir, f"BettiTimeseries_{stat}_ALL_DIMS.png")
        plt.savefig(out_plot, dpi=150)
        plt.close()

        out_csv = os.path.join(self.betti_dir, "summary", "BettiTimeseries_ALL.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return {"timeseries_csv": out_csv, "plot": out_plot}

    def betti_heatmap(self, normalize_t: bool = True) -> Dict[str, Any]:
        dims = self.homology_dimensions[1:]
        entries = self._list_betti_entries()

        df0 = pd.read_csv(entries[0][2])
        t0 = df0["t"].to_numpy(float)
        if normalize_t and t0.size > 1:
            t_plot = (t0 - t0[0]) / (t0[-1] - t0[0])
            t_label = "normalized t"
        else:
            t_plot = t0
            t_label = "t"

        plots = []
        plots_dir = os.path.join(self.betti_dir, "summary", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for k in dims:
            M, years = [], []
            for _, yr, fp in entries:
                df = pd.read_csv(fp)
                hk = df.get(f"H{k}", pd.Series(np.zeros_like(t0))).to_numpy(float)
                M.append(hk)
                years.append(yr)
            M = np.vstack(M) if M else np.zeros((0, t0.size))
            plt.figure()
            extent = (
                [t_plot[0], t_plot[-1], years[0], years[1 if len(years) > 1 else 0]]
                if len(years) > 1
                else [t_plot[0], t_plot[-1], years[0] - 0.5, years[0] + 0.5]
            )
            plt.imshow(M, aspect="auto", origin="lower", extent=extent)
            plt.xlabel(t_label)
            plt.ylabel("Year")
            plt.title(f"Betti heatmap (H{k})")
            plt.tight_layout()
            outp = os.path.join(plots_dir, f"BettiHeatmap_H{k}.png")
            plt.savefig(outp, dpi=150)
            plt.close()
            plots.append(outp)

        return {"plots": plots}

    def betti_peaks_and_auc(self):
        dims = self.homology_dimensions[1:]
        entries = self._list_betti_entries()

        for dim in dims: 

            rows: List[Dict[str, Any]] = []
            xs_year: List[int] = []
            ys_beta_max: List[float] = []
            ys_t_star: List[float] = []
            ys_auc: List[float] = []
            col_h = f"H{dim}"
            for idx, yr, fp in entries:
                df = pd.read_csv(fp)
                t = df["t_norm"].to_numpy(float)
                beta = df.get(col_h, pd.Series(np.zeros_like(t))).to_numpy(float)
                j = int(np.argmax(beta))
                beta_max = float(beta[j])
                t_star = float(t[j])
                auc = float(_trapezoid(beta, t))
                rows.append({"index": idx, "year": yr, "dim": dim, "beta_max": beta_max, "t_star": t_star, "auc": auc})
                xs_year.append(yr)
                ys_beta_max.append(beta_max)
                ys_t_star.append(t_star)
                ys_auc.append(auc)
       
            out_dir = os.path.join(self.betti_dir, "summary")
            plots_dir = os.path.join(out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"BettiPeaks_H{dim}.csv")
            pd.DataFrame(rows, columns=["index", "year", "dim", "beta_max", "t_star", "auc"]).to_csv(out_csv, index=False)

            out_plots: Dict[str, str] = {}
            plt.figure()
            plt.plot(xs_year, ys_beta_max)
            plt.xlabel("Year")
            plt.ylabel(rf"$\beta_{{{dim}}}^{{\max}}$")
            plt.title(f"Betti max over time (H{dim})")
            plt.tight_layout()
            out1 = os.path.join(plots_dir, f"BettiPeaks_beta_max_H{dim}.png")
            plt.savefig(out1, dpi=150)
            plt.close()
            out_plots["beta_max"] = out1

            plt.figure()
            plt.plot(xs_year, ys_t_star)
            plt.xlabel("Year")
            plt.ylabel("t* (normalized)")
            plt.title(f"Argmax location over time (H{dim})")
            plt.tight_layout()
            out2 = os.path.join(plots_dir, f"BettiPeaks_t_star_H{dim}.png")
            plt.savefig(out2, dpi=150)
            plt.close()
            out_plots["t_star"] = out2

            plt.figure()
            plt.plot(xs_year, ys_auc)
            plt.xlabel("Year")
            plt.ylabel("AUC")
            plt.title(f"Betti AUC over time (H{dim})")
            plt.tight_layout()
            out3 = os.path.join(plots_dir, f"BettiPeaks_auc_H{dim}.png")
            plt.savefig(out3, dpi=150)
            plt.close()

        return 

   