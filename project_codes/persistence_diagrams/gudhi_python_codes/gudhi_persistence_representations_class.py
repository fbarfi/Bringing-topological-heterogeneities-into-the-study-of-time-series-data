import os
import pandas as pd
import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0; both compute the same rule.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import gudhi
import matplotlib.pyplot as plt
from gudhi.representations import PersistenceImage
from gudhi.representations.vector_methods import Entropy, Landscape
from gudhi.representations.preprocessing import Padding
from sklearn.preprocessing import MinMaxScaler

class GudhiPersistenceRepresentationsClass:
    def __init__(self, args):
        self.args = args
        self.beginYear  = args['beginYear']
        self.endYear = args['endYear']
        self.n_years = list(args['n_years'])
        self.data_file = '../../project_data/dataset_augm.csv'
        self.homology_dimensions = args['homology_dimensions']
        self.Hidx = int(''.join(map(str, self.homology_dimensions[1:])))
        self.folder = args['folder']
        self.images = args['images']
        self.C = args['C']
        self.p = args['p']
        self.bins = args['bins']
        self.max_edge_length = args['max_edge_length']
        self.max_dimension = args['max_dimension']
        self.n_components = args['n_components']
        self.scaler = MinMaxScaler()
        self.embedding_file = os.path.join(self.folder, "embedded_points.npz")

        emb_path = os.path.join(self.folder, "embedded_points.npz")
        if not os.path.exists(emb_path):
            raise FileNotFoundError("Embedded points missing. Run GudhiPersistenceDiagramsClass.grouping_PDs() first.")
        self.embedded_points = np.load(emb_path, allow_pickle=True)
        self.year_to_index = {year: i for i, year in enumerate(self.n_years)}

    # -------- data ----------
    def data_preprocessing(self):
        df_y = pd.read_csv(self.data_file, usecols=['year','ccode','revonsets'])
        df_X = pd.read_csv(self.data_file).drop(columns=['revonsets'])
        return df_X, df_y

    # -------- load PDs (H1 and beyond) ----------
    def upload_PDs_by_dim(self):
        """
        Returns dict: {dim: [diagram_per_year]} for requested dims (default H1–H5).
        Expects files: PersistenceDiagrams/{filename_prefix}_{idx}.csv with columns b,d,h_dim.
        """
        filename_prefix=f"PD_H_{self.Hidx}"
        dims = self.homology_dimensions

        dfs = []
        for idx in range(len(self.n_years)):
            path = os.path.join(self.folder, f'PersistenceDiagrams/{filename_prefix}_{idx}.csv')
            df_i = pd.read_csv(path)
            need = {'b','d','h_dim'}
            if not need.issubset(df_i.columns):
                raise ValueError(f"{path} must have columns {need}")
            dfs.append(df_i)

        by_dim = {k: [] for k in dims}
        for df_i in dfs:
            groups = dict(tuple(df_i.groupby('h_dim')))
            for k in dims:
                arr = groups[k][['b','d']].to_numpy(dtype=float) if k in groups else np.zeros((0,2), float)
                by_dim[k].append(arr)

        for k in dims:
            by_dim[k] = Padding().fit_transform(by_dim[k])
        return by_dim

    # -------- Entropy ---------
    def persistence_entropy(self, dims=None):
        dims = self.homology_dimensions[1:]
        pd_by_dim = self.upload_PDs_by_dim()

        pe = Entropy()
        out = {"year": self.n_years}
        for k in dims:
            
            vals = np.array(pe.fit_transform(pd_by_dim[k]), dtype=float).reshape(-1)
            mx = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
            out[f"H_{k}"] = vals / (mx if mx != 0 else 1.0)
        df = pd.DataFrame(out)
        df.to_csv(os.path.join(self.folder, f"Entropy_H_{self.Hidx}.csv"), index=False)
        return df

    # -------- Amplitude (H1 and beyond) ----------
    def amplitude(self, dims=None, p=2.0):
        dims = self.homology_dimensions[1:]
        pd_by_dim = self.upload_PDs_by_dim()

        out = {"year": self.n_years}
        p_eff = float(p)
        for k in dims:
            seq = []
            for diag in pd_by_dim[k]:
                if diag.size == 0:
                    seq.append(0.0); continue
                b, d = diag[:,0], diag[:,1]
                m = np.isfinite(b) & np.isfinite(d) & (d > b)
                pers = (d[m] - b[m]) / 2.0
                if pers.size == 0:
                    seq.append(0.0)
                elif np.isinf(p_eff):
                    seq.append(float(np.max(pers)))
                else:
                    seq.append(float(np.linalg.norm(pers, ord=p_eff)))
            out[f"H_{k}"] = seq

        df = pd.DataFrame(out)
        df.to_csv(os.path.join(self.folder, f"Amplitude_H_{self.Hidx}.csv"), index=False)
        return df

    # -------- Number of points (H0 and beyond ) ----------
    def number_of_points(self):
        dims = self.homology_dimensions
        hidx = int(''.join(map(str, dims)))

        pd_by_dim = self.upload_PDs_by_dim()    

        out = {"year": self.n_years}
        for k in dims:
            counts = []
            for diag in pd_by_dim[k]:
                if diag.size == 0:
                    counts.append(0); continue
                b, d = diag[:,0], diag[:,1]
                m = np.isfinite(b) & np.isfinite(d) & (d > b)
                counts.append(int(np.sum(m)))
            out[f"H_{k}"] = counts

        df = pd.DataFrame(out)
        df.to_csv(os.path.join(self.folder, f"NumberOfPoints_H_{hidx}.csv"), index=False)
        return df

    # -------- Landscapes + L2 (H1 and beyond) ----------
    def landscape_l2(self, dims=None, num_landscapes=5, resolution=100):
        dims =  self.homology_dimensions[1:]
        pd_by_dim = self.upload_PDs_by_dim()

        landscape_dir = os.path.join(self.folder, "PersistenceLandscapes")
        os.makedirs(landscape_dir, exist_ok=True)

        l2_df = {"year": self.n_years}
        for k in dims:
            land = Landscape(num_landscapes=int(num_landscapes), resolution=int(resolution))
            land.fit(pd_by_dim[k])
            V = land.transform(pd_by_dim[k])
            grid = np.asarray(land.grid_, dtype=float)
            step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0

            n = V.shape[0]
            V3 = V.reshape(n, int(num_landscapes), int(resolution))
            l2_vals = np.sqrt(np.sum(V3 * V3, axis=(1, 2)) * step)
            l2_df[f"L2_landscape_H{k}"] = l2_vals

            for i in range(n):
                data = {"t": grid}
                for ell in range(int(num_landscapes)):
                    data[f"lambda{ell+1}"] = V3[i, ell, :]
                pd.DataFrame(data).to_csv(
                    os.path.join(landscape_dir, f"landscape_H{k}_{i}.csv"),
                    index=False
                )

        pd.DataFrame({"index": np.arange(len(self.n_years), dtype=int), "year": self.n_years}).to_csv(
            os.path.join(landscape_dir, "index.csv"), index=False
        )
        pd.DataFrame(l2_df).to_csv(os.path.join(self.folder, f"L2_Landscape_H_{self.Hidx}.csv"), index=False)
        return

    # -------- C1 from saved landscapes (H1 and beyond) ----------
    def compute_c1(self, dims=None):
        dims = self.homology_dimensions[1:]
        landscape_dir = os.path.join(self.folder, "PersistenceLandscapes")
        idx_map_path = os.path.join(landscape_dir, "index.csv")
        if not os.path.exists(idx_map_path):
            raise FileNotFoundError(f"Missing index file: {idx_map_path}")
        idx_map = pd.read_csv(idx_map_path)
        if "index" not in idx_map.columns:
            raise ValueError("index.csv must contain an 'index' column")
        years = idx_map["year"].tolist() if "year" in idx_map.columns else None
        indices = idx_map["index"].astype(int).tolist()

        out_L1 = {"index": indices}
        if years is not None:
            out_L1["year"] = years

        out_C1 = out_L1.copy()

        for k in dims:
            L1_vals = []
            for i in indices:
                f = os.path.join(landscape_dir, f"landscape_H{k}_{i}.csv")
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Missing landscape file: {f}")
                df = pd.read_csv(f)

                if "t" in df.columns and len(df["t"]) > 1:
                    t = df["t"].to_numpy(dtype=float)
                    diffs = np.diff(t)
                    diffs = diffs[np.isfinite(diffs)]
                    step = float(np.median(diffs)) if diffs.size else 1.0
                else:
                    step = 1.0

                lambda_cols = [c for c in df.columns if c.startswith("lambda")]
                vals = df[lambda_cols].to_numpy(dtype=float) if lambda_cols else np.zeros((0,0), float)
                L1_vals.append(float(np.sum(np.abs(vals)) * step) if vals.size else 0.0)

            L1_arr = np.asarray(L1_vals, dtype=float)
            C1_arr = np.empty_like(L1_arr)
            if L1_arr.size:
                C1_arr[0] = L1_arr[0]
                if L1_arr.size > 1:
                    C1_arr[1:] = L1_arr[1:] + np.abs(L1_arr[1:] - L1_arr[:-1])

            out_L1[f"L1_landscape_H{k}"] = L1_arr
            out_C1[f"C1_landscape_H{k}"] = C1_arr

        pd.DataFrame(out_L1).to_csv(os.path.join(self.folder, f"L1_Landscape_H_{self.Hidx}.csv"), index=False)
        pd.DataFrame(out_C1).to_csv(os.path.join(self.folder, f"C1_Landscape_H_{self.Hidx}.csv"), index=False)
        return

    

    def betti_numbers(self, dims=None, resolution=200, make_plots=True, save_curves=True):
        """
        Compute Betti curves β_k(t) for H1–H4 across all years.
        Save ONE file per year with columns: index, year, t, H1, H2, H3, H4.
        Also saves per-dimension plots (optional).
        """
        dims = list(self.homology_dimensions[1:]) 
        if not dims:
            raise ValueError("No valid homology dimensions selected; choose from {1,2,3,4}.")

        pd_by_dim = self.upload_PDs_by_dim()

        out_dir = os.path.join(self.folder, "BettiNumbers")
        plot_dir = os.path.join(out_dir, "plots")
        os.makedirs(out_dir, exist_ok=True)
        if make_plots:
            os.makedirs(plot_dir, exist_ok=True)

        all_b, all_d = [], []
        for k in dims:
            for diag in pd_by_dim[k]:
                if diag.size:
                    b, d = diag[:, 0], diag[:, 1]
                    m = np.isfinite(b) & np.isfinite(d) & (d > b)
                    if np.any(m):
                        all_b.append(b[m]); all_d.append(d[m])
        if all_b and all_d:
            min_b = float(np.min([np.min(x) for x in all_b]))
            max_d = float(np.max([np.max(x) for x in all_d]))
        else:
            min_b, max_d = 0.0, 1.0
        if not np.isfinite(min_b): min_b = 0.0
        if not np.isfinite(max_d): max_d = 1.0
        if max_d <= min_b:
            eps = max(1e-6, abs(max_d - min_b) + 1e-6)
            max_d = min_b + eps

        t = np.linspace(min_b, max_d, int(resolution))

        pd.DataFrame(
                {"index": np.arange(len(self.n_years), dtype=int), "year": self.n_years}
            ).to_csv(os.path.join(out_dir, "index.csv"), index=False)

        # β-curve helper
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
            trow = grid[None, :]                 # (1, G)
            alive = (b <= trow) & (trow < d)     # (k, G)
            return alive.sum(axis=0).astype(int)

        curves_by_dim = {}
        for k in dims:
            curves_by_dim[k] = [ _betti_curve(diag, t) for diag in pd_by_dim[k] ]

            if make_plots and len(curves_by_dim[k]) > 0:
                plt.figure()
                for i, beta in enumerate(curves_by_dim[k]):
                    lbl = str(self.n_years[i]) if i < len(self.n_years) else f"idx {i}"
                    plt.plot(t, beta, label=lbl)
                plt.xlabel("Filtration value (t)")
                plt.ylabel(rf"Betti $\beta_{{{k}}}(t)$")
                plt.title(f"Betti Curves (H{k})")
                if len(curves_by_dim[k]) <= 15:
                    plt.legend(loc="best", fontsize="small")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"betti_H{k}.png"), dpi=200)
                plt.close()

        if save_curves:
            n_years = len(self.n_years)
            for i in range(n_years):
                row = {
                    "index": np.full_like(t, i, dtype=int),
                    "year":  np.full_like(t, self.n_years[i], dtype=int),
                    "t":     t.astype(float),
                    "H1":    curves_by_dim.get(1, [np.zeros_like(t)])[i] if 1 in dims else np.zeros_like(t),
                    "H2":    curves_by_dim.get(2, [np.zeros_like(t)])[i] if 2 in dims else np.zeros_like(t),
                    "H3":    curves_by_dim.get(3, [np.zeros_like(t)])[i] if 3 in dims else np.zeros_like(t),
                    "H4":    curves_by_dim.get(4, [np.zeros_like(t)])[i] if 4 in dims else np.zeros_like(t),
                }
                df_year = pd.DataFrame(row, columns=["index","year","t","H1","H2","H3","H4"])
                df_year.to_csv(os.path.join(out_dir, f"betti_{self.Hidx}_{i}.csv"), index=False)

        return


    # -------- Betti summaries (area, mean, max, l2) per dim/year ----------
    def betti_summary(self, dims=None, metrics=("area", "mean", "max", "l2"), ensure_curves=True):
        """
        Compute aggregates from saved Betti curves and write:
          - {folder}/BettiNumbers/summary/BettiSummary_H{k}.csv (per dim)
          - {folder}/BettiNumbers/summary/BettiSummary_ALL.csv (all dims)
          - {folder}/BettiNumbers/summary/plots/BettiSummary_H{k}.png (overlay lines per metric)

        Why: enables compact comparisons across years without scanning full curves.
        """

        if dims is None:
            dims = list(self.homology_dimensions[1:])
        if not dims:
            raise ValueError("No valid homology dimensions selected; choose from {1,2,3,4}.")

        out_dir = os.path.join(self.folder, "BettiNumbers")
        idx_path = os.path.join(out_dir, "index.csv")
        if ensure_curves and not os.path.exists(idx_path):
            self.betti_numbers(dims=dims, make_plots=False, save_curves=True)

        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing Betti index: {idx_path}")
        idx_df = pd.read_csv(idx_path)
        if "index" not in idx_df.columns:
            raise ValueError("BettiNumbers/index.csv must contain 'index' column")
        years = idx_df["year"].tolist() if "year" in idx_df.columns else list(range(len(idx_df)))

        summary_dir = os.path.join(out_dir, "summary")
        # plots_dir = os.path.join(summary_dir, "plots")
        os.makedirs(summary_dir, exist_ok=True)
        # os.makedirs(plots_dir, exist_ok=True)

        all_rows = []

        def _safe_span(t_arr: np.ndarray) -> float:
            if t_arr.size < 2:
                return 0.0
            a, b = t_arr[0], t_arr[-1]
            if not (np.isfinite(a) and np.isfinite(b)):
                return 0.0
            return float(b - a)

        for k in dims:
            rows_k = []
            metric_series = {m: [] for m in metrics}
            for i, yr in enumerate(years):
                # each year now saved in one file with columns: index, year, t, H1..H4
                curve_path = os.path.join(out_dir, f"betti_{self.Hidx}_{i}.csv")
                df = pd.read_csv(curve_path)

                t = df["t"].to_numpy(dtype=float) if "t" in df.columns else np.array([], float)
                hk_col = f"H{k}"
                beta = df[hk_col].to_numpy(dtype=float) if hk_col in df.columns else np.array([], float)

                if t.size == 0 or beta.size == 0:
                    area = mean = l2 = maxv = 0.0
                else:
                    area = float(_trapezoid(beta, t))
                    span = _safe_span(t)
                    mean = float(area / span) if span > 0 else float(np.mean(beta))
                    l2 = float(np.sqrt(_trapezoid(beta * beta, t)))
                    maxv = float(np.max(beta)) if beta.size else 0.0

                row = {"year": yr, "dim": k, "area": area, "mean": mean, "max": maxv, "l2": l2}
                rows_k.append(row)
                all_rows.append(row)
                for m in metrics:
                    metric_series[m].append(row[m])


            plt.figure()
            x = [r["year"] for r in rows_k]
            for m in metrics:
                plt.plot(x, metric_series[m], label=m)  
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.title(f"Betti Summary Metrics (H{k})")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, f"BettiSummary_H{k}.png"), dpi=150)
            plt.close()

        pd.DataFrame(all_rows).to_csv(os.path.join(summary_dir, "BettiSummary_ALL.csv"), index=False)

        
        return 