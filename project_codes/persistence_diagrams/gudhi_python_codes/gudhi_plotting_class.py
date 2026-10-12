import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import gudhi as gd

class GudhiPlottingClass:
    def __init__(self, args):
        self.args = args
        self.dataset_file = '../../project_data/dataset_augm.csv'
        self.folder = args['folder']
        self.images = args['images']
        self.homology_dimensions = args['homology_dimensions']
        self.Hidx = int(''.join(map(str, self.homology_dimensions[1:])))
        self.max_dimension = args['max_dimension']
        self.float_vars = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity','revonsets']
        self.binary_vars = ['revleader', 'new_oil_state', 'coldwar', 'majorpower']
        self.n_years = list(args['n_years'])
        self.years = args['years']    
        self.cases = args['cases']     

    def _pd_path(self, idx):
        return os.path.join(self.folder, f'PersistenceDiagrams/PD_H_{self.Hidx}_{idx}.csv')

    def _ensure_dir(self, p):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    def _dim_grid(self, dims):
        n = len(dims)
        return (2, 2) if n > 2 else (1, n)

    def data_loading(self):
        PDs = [pd.read_csv(self._pd_path(idx)) for idx in range(len(self.n_years))]

        df = pd.read_csv(self.dataset_file).sort_values(['year', 'ccode'])
        df = df.drop(columns=['revonsets'])
        df_X = df[['year', 'ccode'] + self.binary_vars + self.float_vars]

        X_list = [df_X.loc[df_X['year'] == y, :].iloc[:, 2:].reset_index(drop=True).values
                  for y in df_X['year'].unique().tolist()]
        X_list = [np.swapaxes(X, 0, 1) for X in X_list]

        all_diagrams = []
        for dfi in PDs:
            diagram = []
            for _, row in dfi.iterrows():
                dim = int(row['h_dim'])
                b = float(row['b']); d = float(row['d'])
                diagram.append((dim, (b, d)))
            all_diagrams.append(diagram)

        PE_df = pd.read_csv(os.path.join(self.folder, f'Entropy_H_{self.Hidx}.csv'))

        return X_list, all_diagrams, PE_df

    def uploading_PDs(self):
        PDs = []
        for idx in range(len(self.n_years)):
            df = pd.read_csv(os.path.join(self.folder, f'PersistenceDiagrams/PD_H_{self.Hidx}_{idx}.csv'))
            diag = []
            for _, row in df.iterrows():
                dim = int(row['h_dim'])
                b = float(row['b'])
                d = float(row['d'])
                if 1 <= dim <= 5:
                    diag.append((dim, (b, d)))
            PDs.append(diag)
        return PDs


    def Plotting_Persistence_Diagrams(self):
        chosen_years = self.years
        cases = self.cases

        PDs = self.uploading_PDs()  

        max_dim = 0
        for diag in PDs:
            if diag:
                max_dim = max(max_dim, max(dim for dim, _ in diag))
        base_colors = [
            (0.8, 0.8, 0.8),  # dim 0 (unused)
            (0.9, 0.1, 0.1),  # H1
            (0.1, 0.5, 0.9),  # H2
            (0.1, 0.7, 0.1),  # H3
            (0.60, 0.20, 0.70),  # H4
            (0.70, 0.30, 0.60),  # H5
        ]
        if len(base_colors) < (max_dim + 1):
            base_colors += [base_colors[-1]] * ((max_dim + 1) - len(base_colors))

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()

        for i, idx in enumerate(cases):
            ax = axs[i]
            diag = PDs[idx]
            if not diag:
                ax.set_title(f"{chosen_years[i]} (no points)")
                ax.set_xlabel("Birth"); ax.set_ylabel("Death"); ax.grid(False)
                continue

            gd.plot_persistence_diagram(
                diag, axes=ax, greyblock=False, alpha=0.6, colormap=tuple(base_colors)
            )
            ax.set_title(f"{chosen_years[i]}")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.images, "Persistence_Diagrams_4_years.png"))
        plt.close()


    def Plotting_Persistence_Barcodes(self):
        PDs = self.uploading_PDs()
        pds = [PDs[i] for i in self.cases]
        chosen_years = self.years

        base_colors = [
            (0.8, 0.8, 0.8),
            (0.9, 0.1, 0.1),
            (0.1, 0.5, 0.9),
            (0.1, 0.7, 0.1),
            (0.60, 0.20, 0.70),
            (0.70, 0.30, 0.60), 
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for year, ax, diag in zip(chosen_years, axes, pds):
            if not diag:
                ax.set_title(f'{year} (no points)')
                continue
            gd.plot_persistence_barcode(diag, axes=ax, colormap=tuple(base_colors))
            ax.set_title(f'{year}', fontsize=15)

        plt.suptitle('Persistence Barcodes', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images, 'Persistence_Barcodes_4_years.png'))
        plt.close()


    def Plotting_Entropy(self, dims=None):
        dims = self.homology_dimensions[1:] if dims is None else list(dims)
        df = pd.read_csv(os.path.join(self.folder, f'Entropy_H_{self.Hidx}.csv'))
        rows, cols = self._dim_grid(dims)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6) if rows==1 else (12, 10))
        axes = np.array(axes).reshape(rows, cols)

        for j, k in enumerate(dims):
            r, c = divmod(j, cols)
            y = df[f'H_{k}'].to_numpy(dtype=float)
            x = df['year'].to_numpy(dtype=int)
            ax = axes[r, c]
            ax.plot(x, y); ax.scatter(x, y)
            ax.set_title(f'Persistence Entropy – H{k}')
            ax.set_xlabel('Year'); ax.set_ylabel('Entropy')

        plt.tight_layout()
        self._ensure_dir(os.path.join(self.images, "x.png"))
        plt.savefig(os.path.join(self.images, f'Persistence_Entropy_H_{self.Hidx}.png'), dpi=300)
        plt.close()

    def Plotting_Amplitude(self, dims=None):
        dims = self.homology_dimensions[1:] if dims is None else list(dims)
        df = pd.read_csv(os.path.join(self.folder, f'Amplitude_H_{self.Hidx}.csv'))
        rows, cols = self._dim_grid(dims)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6) if rows==1 else (12, 10))
        axes = np.array(axes).reshape(rows, cols)

        for j, k in enumerate(dims):
            r, c = divmod(j, cols)
            y = df[f'H_{k}'].to_numpy(dtype=float)
            x = df['year'].to_numpy(dtype=int)
            ax = axes[r, c]
            ax.plot(x, y); ax.scatter(x, y)
            ax.set_title(f'Amplitude – H{k}')
            ax.set_xlabel('Year'); ax.set_ylabel('Amplitude')

        plt.tight_layout()
        self._ensure_dir(os.path.join(self.images, "x.png"))
        plt.savefig(os.path.join(self.images, f'Amplitude_H{self.Hidx}.png'), dpi=300)
        plt.close()

    def Plotting_NumberOfPoints(self):
       
        dims = self.homology_dimensions 
        hidx = int(''.join(map(str, dims)))

        df_path = os.path.join(self.folder, f'NumberOfPoints_H_{hidx}.csv')
        df = pd.read_csv(df_path)

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flatten()

        years = df['year'].to_numpy(dtype=int)

        with mpl.rc_context({"text.usetex": False}):
            for k, ax in enumerate(axes):
                if k < 5:
                    y = df[f'H_{k}'].to_numpy(dtype=float)
                    ax.plot(years, y)
                    ax.scatter(years, y)
                    ax.set_title(f'Number of Points – H{k}')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of Points')

            # hide any unused subplots
            axes[-1].remove()

            plt.tight_layout()
            self._ensure_dir(os.path.join(self.images, "x.png"))
            plt.savefig(os.path.join(self.images, f'NumberOfPoints_H_{hidx}.png'), dpi=200)
            plt.close()


    def Plotting_landscapes(self, dim=1):
        years = self.years
        cases = self.cases
        landscape_dir = self.args["PersistenceLandscapes"]
        out_png = os.path.join(self.images, f"Persistence_Landscapes_4_years_H{dim}.png")

        custom_colors = [
            (0.9, 0.1, 0.1),
            (0.1, 0.5, 0.9),
            (0.1, 0.7, 0.1),
            (0.60, 0.20, 0.70),
            (0.95, 0.70, 0.15),
        ]

        with mpl.rc_context({"text.usetex": False}):
            fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
            axs = axs.ravel()

            for i, idx in enumerate(cases):
                ax = axs[i]
                csv_path = os.path.join(landscape_dir, f"landscape_H{dim}_{idx}.csv")
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"Missing file: {csv_path}")

                df_land = pd.read_csv(csv_path)
                if "t" not in df_land.columns:
                    raise ValueError(f"`t` column not found in {csv_path}")

                lambda_cols = [c for c in df_land.columns if c.startswith("lambda")]
                Lp = len(lambda_cols)
                x = df_land["t"].to_numpy()
                for ell in range(Lp):
                    y = df_land[lambda_cols[ell]].to_numpy()
                    ax.plot(x, y, color=custom_colors[ell % len(custom_colors)], linewidth=1.6, alpha=0.9,
                            label=fr"$\lambda_{{{ell+1}}}$")

                ax.set_title(f"H{dim} Landscape · year={years[i]}")
                ax.set_xlabel("t"); ax.set_ylabel(r"$\lambda_k(t)$")
                ax.legend()

        fig.tight_layout()
        self._ensure_dir(out_png)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

    def plot_l2_series(self, dim=1):
        hidx = self.homology_dimensions[1:] 
        l2_csv = os.path.join(self.folder, f"L2_Landscape_H_{self.Hidx}.csv")
        out_png = os.path.join(self.images, f"L2_landscape_H{dim}.png")

        df = pd.read_csv(l2_csv)
        years = df["year"].to_numpy()
        l2_vals = df[f"L2_landscape_H{dim}"].to_numpy()

        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            fig = plt.figure(figsize=(8, 4))
            plt.plot(years, l2_vals)
            plt.title(fr"$L^2$ norm of $H_{dim}$ landscape")
            plt.xlabel("year"); plt.ylabel(r"$\|\lambda\|_2$")
            fig.tight_layout()
            self._ensure_dir(out_png)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

    def plot_c1_series(self, dim=1):
        c1_csv = os.path.join(self.folder, f"C1_Landscape_H_{self.Hidx}.csv")
        out_png = os.path.join(self.images, f"C1_landscape_H{dim}.png")

        df = pd.read_csv(c1_csv)
        if "year" in df.columns:
            x = df["year"].to_numpy(); xlabel = "year"
        else:
            x = df["index"].to_numpy(); xlabel = "index"

        y = df[f"C1_landscape_H{dim}"].to_numpy()

        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            fig = plt.figure(figsize=(9, 4))
            plt.plot(x, y, linewidth=1.6)
            plt.title(fr"$C^1$-norm of $H_{dim}$ landscape")
            plt.xlabel(xlabel); plt.ylabel(r"$\|\lambda\|_{C^1}$")
            plt.tight_layout()
            self._ensure_dir(out_png)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

    def plot_c1_l1_series(self, dim=1):
        c1_csv = os.path.join(self.folder, f"C1_Landscape_H_{self.Hidx}.csv")
        l1_csv = os.path.join(self.folder, f"L1_Landscape_H_{self.Hidx}.csv")
        out_png = os.path.join(self.images, f"C1_L1_landscape_H{dim}.png")

        c1_color = (0.9, 0.1, 0.1)
        l1_color = (0.1, 0.5, 0.9)
        linewidth = 1.8

        C1_df = pd.read_csv(c1_csv)
        L1_df = pd.read_csv(l1_csv)
        if "year" in C1_df.columns:
            x = C1_df["year"].to_numpy(); xlabel = "year"
        else:
            x = C1_df["index"].to_numpy(); xlabel = "index"

        c1 = C1_df[f"C1_landscape_H{dim}"].to_numpy(dtype=float)
        l1 = L1_df[f"L1_landscape_H{dim}"].to_numpy(dtype=float)

        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            fig, ax = plt.subplots(figsize=(10, 4.5))
            l_c1, = ax.plot(x, c1, color=c1_color, linewidth=linewidth, label=r"$C^1$")
            ax.set_xlabel(xlabel); ax.set_ylabel(r"$C^1$", color=c1_color)
            ax.tick_params(axis='y', labelcolor=c1_color)

            ax2 = ax.twinx()
            l_l1, = ax2.plot(x, l1, color=l1_color, linewidth=linewidth, linestyle="--", label=r"$L^1$")
            ax2.set_ylabel(r"$L^1$", color=l1_color)
            ax2.tick_params(axis='y', labelcolor=l1_color)

            lines = [l_c1, l_l1]
            labels = [ln.get_label() for ln in lines]
            ax.legend(lines, labels, loc="upper left")
            plt.title(fr"$C^1$ and $L^1$ norms of $H_{dim}$ landscape")
            plt.tight_layout()
            self._ensure_dir(out_png)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)

    def plot_c1_l1_l2(self, dim=1):
        c1_csv = os.path.join(self.folder, f"C1_Landscape_H_{self.Hidx}.csv")
        l1_csv = os.path.join(self.folder, f"L1_Landscape_H_{self.Hidx}.csv")
        l2_csv = os.path.join(self.folder, f"L2_Landscape_H_{self.Hidx}.csv")
        out_png = os.path.join(self.images, f"C1_L1_L2_landscape_H_{dim}.png")

        df_c1 = pd.read_csv(c1_csv)
        df_l1 = pd.read_csv(l1_csv)
        df_l2 = pd.read_csv(l2_csv)

      
        df = pd.concat(
            [
            df_c1[["year", f"C1_landscape_H{dim}"]],
            df_l1[f"L1_landscape_H{dim}"],
            df_l2[f"L2_landscape_H{dim}"]
            ],
            axis=1).sort_values("year")
        
        x = df["year"].to_numpy(); xlabel = "year"

        c1 = df[f"C1_landscape_H{dim}"].to_numpy(dtype=float)
        l1 = df[f"L1_landscape_H{dim}"].to_numpy(dtype=float)
        l2 = df[f"L2_landscape_H{dim}"].to_numpy(dtype=float)


        c_c1 = (0.90, 0.10, 0.10)
        c_l1 = (0.10, 0.50, 0.90)
        c_l2 = (0.10, 0.70, 0.10)

        with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
            fig, ax = plt.subplots(figsize=(11, 5))
            ln1, = ax.plot(x, c1, color=c_c1, linewidth=1.8, label="C1")
            ax.set_ylabel("C1", color=c_c1); ax.tick_params(axis="y", labelcolor=c_c1)

            ax2 = ax.twinx()
            ln2, = ax2.plot(x, l1, color=c_l1, linewidth=1.8, linestyle="--", label="L1")
            ax2.set_ylabel("L1", color=c_l1); ax2.tick_params(axis="y", labelcolor=c_l1)

            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("outward", 50))
            ax3.spines["right"].set_visible(True)
            ln3, = ax3.plot(x, l2, color=c_l2, linewidth=1.8, linestyle=":", label="L2")
            ax3.set_ylabel("L2", color=c_l2); ax3.tick_params(axis="y", labelcolor=c_l2)

            ax.set_xlabel(xlabel); ax.set_title(f"C1, L1, and L2 (H{dim})")
            lines = [ln1, ln2, ln3]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper left")
            fig.tight_layout()
            self._ensure_dir(out_png)
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


    def plot_covariates_overlaid_with_composite_pca1(self):

        def _load_pc1_csv(path: str) -> pd.DataFrame:
            df = pd.read_csv(path)
            out = df[["year", "pc1"]].copy()
            out["year"] = out["year"].astype(int)
            out["pc1"] = out["pc1"].astype(float)
            return out.sort_values("year")

        def _z(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float)
            mu = np.nanmean(a); sd = np.nanstd(a)
            return (a - mu) / (sd + 1e-12)

        def _composite_path_for_dim(k: int) -> str | None:
            guess = os.path.join(self.folder, f"Composite_H{k}_merged_wide.csv")
            return guess if os.path.exists(guess) else None

        # per-year covariate means (binary->proportions)
        df_all = pd.read_csv(self.dataset_file).sort_values(["year", "ccode"])  
        covars = self.binary_vars + self.float_vars
        cov_year_all = df_all.groupby("year")[covars].mean().reset_index()

        dims = list(self.homology_dimensions[1:]) 
        results: dict[str, dict] = {}

        for k in dims:
            path_to_use = _composite_path_for_dim(k)
            comp = _load_pc1_csv(path_to_use)

            years = np.intersect1d(comp["year"].to_numpy(int), cov_year_all["year"].to_numpy(int))
            comp_k = comp[comp["year"].isin(years)].sort_values("year")
            cov_year = cov_year_all[cov_year_all["year"].isin(years)].sort_values("year")
            yvec = comp_k["year"].to_numpy(int)
            pc1_z = _z(comp_k["pc1"].to_numpy(float))

            used_anchor = None
            used_corr = np.nan
            used_sign = +1

            best_name, best_val = None, -np.inf
            for var in covars:
                v = _z(cov_year[var].to_numpy(float))
                r = float(np.corrcoef(pc1_z, v)[0, 1]) if np.isfinite(v).all() else np.nan
                if np.isfinite(r) and abs(r) > best_val:
                    best_val = abs(r)
                    best_name = var
            anchor_k = best_name

            a_z = _z(cov_year[anchor_k].to_numpy(float))
            r = float(np.corrcoef(pc1_z, a_z)[0, 1]) if np.isfinite(a_z).all() else np.nan
            if np.isfinite(r) and r < 0:
                pc1_z = -pc1_z
                used_sign = -1
                used_corr = -r
            else:
                used_corr = r
            used_anchor = anchor_k

            fig, axes = plt.subplots(4, 3, figsize=(16, 12), sharex=True)
            axes = axes.flatten() 

            with mpl.rc_context({"text.usetex": False, "axes.unicode_minus": False}):
                for i, var in enumerate(covars):
                    ax = axes[i]
                    cov_series = cov_year[var].to_numpy(float)
                    ln_cov, = ax.plot(yvec, cov_series, linewidth=1.7, label=var)
                    ax.set_title(var)
                    ax.set_xlabel("Year")
                    ax.set_ylabel(var)

                    ax2 = ax.twinx()
                    ln_pc1, = ax2.plot(yvec, pc1_z, linewidth=1.3, linestyle="--", label="Composite PCA1")
                    ax2.set_ylabel("PCA1")

                    lines = [ln_cov, ln_pc1]
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc="upper left", fontsize=8)
                    ax.grid(False)

                # for i in [1,2]:
                axes[-1].remove()

                suffix = f" · anchor={used_anchor}, sign={used_sign:+d}" if used_anchor else ""
                fig.suptitle(f"Covariates (orig units) vs Composite PCA-1 (z) · H{k}{suffix}", fontsize=14)
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            out_png = os.path.join(self.images, f"Covariates_vs_CompositePCA1_H{k}.png")
            self._ensure_dir(out_png)
            fig.savefig(out_png, dpi=200)
            plt.close(fig)

        return 

    def plot_midpoint_hist(self):
        max_bins=120
        dims = self.homology_dimensions[1:]
        y1, y2 = 2000, 2010

        def collect_mids_for_year(year, dims):
            years_list = list(self.n_years)
            year_idx = years_list.index(year)
            PDs = self.uploading_PDs()
            diag = PDs[year_idx]
            mids = {d: [] for d in dims}
            for dim, bd in diag:
                b, d = bd
                if dim in dims:
                    mids[int(dim)].append(0.5 * (float(b) + float(d)))
            for d in dims:
                mids[d] = np.asarray(mids[d], dtype=float)
            return mids


        fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharey="row")  # axs is 2x4
        for r, y in enumerate((y1, y2)):
            mids = collect_mids_for_year(y, dims) 
            for c, dim in enumerate(dims):
                ax = axs[r, c]
                ax.set_title(f"{y} – H{dim}")
                ax.set_xlabel("(b + d) / 2")
                if c == 0:
                    ax.set_ylabel("Density")
                ax.grid(False)
                data = mids.get(dim, np.array([]))
                ax.hist(data, bins=20, density=True, edgecolor="black", alpha=0.6)

        plt.suptitle(" Histograms for  H in [1,2,3,4] for Years = [2000,2010]")
        plt.tight_layout()
    
    
    # def plot_midpoint_hist(self):

    #     dims = list(self.homology_dimensions[1:])
    #     y1, y2 = 2000, 2010

    #     def collect_mids_for_year(year, dims):
    #         years_list = list(self.n_years)
    #         year_idx = years_list.index(year)
    #         PDs = self.uploading_PDs()
    #         diag = PDs[year_idx]
    #         mids = {d: [] for d in dims}
    #         for dim, (b, d) in diag:
    #             if dim in dims:
    #                 mids[int(dim)].append(0.5 * (float(b) + float(d)))
    #         for d in dims:
    #             mids[d] = np.asarray(mids[d], dtype=float)
    #         return mids

    #     fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    #     for r, y in enumerate((y1, y2)):
    #         mids = collect_mids_for_year(y, dims)
    #         for dim in dims:
    #             ax = axs[r, dim-1]
    #             sns.histplot(mids[dim], bins=30, stat="density", ax=ax)
    #             ax.set_title(f"{y} – H{dim}")
    #             ax.set_xlim(0.5, 1.0) 
    #             ax.set_xlabel("(b + d) / 2")
    #             ax.set_ylabel("Density")
               
    #     plt.suptitle("Histograms of (b+d)/2 for H∈{1,2,3,4} in 2000 & 2010")
    #     plt.tight_layout()

        fname = f"Midpoint_Hist_Compare_{y1}_{y2}.png"
        plt.savefig(os.path.join(self.images, fname), dpi=200)
        plt.close()
        
