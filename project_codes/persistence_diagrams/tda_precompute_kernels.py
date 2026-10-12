import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import gc

from gudhi.representations.preprocessing import Padding
from gudhi.wasserstein import wasserstein_distance

# ---- EDIT CONFIG
CONFIG = {
    "pd_folder": "../../project_results/gudhi_multiple_tda_results_H_4/PersistenceDiagrams",
    "cache_dir": "../../project_results/gudhi_multiple_tda_results_H_4/gpd/pd_kernel_cache_all",
    "images_dir": "../../project_results/gudhi_multiple_tda_results_H_4/images/",
    "years": (1945, 2011),
    "dims": (1, 2, 3, 4),
    "Hidx": "1234",
    "sigmas": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    "dtype_disk": "float32",
}


def nearest_psd_correlation(K, min_eig=1e-8):
    K = np.asarray(K, dtype="float64")
    K = 0.5 * (K + K.T)
    w, Q = np.linalg.eigh(K)
    w = np.clip(w, min_eig, None)
    K_psd = (Q * w) @ Q.T
    d = np.sqrt(np.clip(np.diag(K_psd), min_eig, None))
    K_corr = (K_psd / d[:, None]) / d[None, :]
    K_corr = 0.5 * (K_corr + K_corr.T)
    np.fill_diagonal(K_corr, 1.0)
    return K_corr


def sanitize_diagram(arr):
    """
    Remove non-finite rows, enforce d>=b, 
    ensure C-contiguous, keep at least one point.
    """

    A = np.asarray(arr, dtype="float64", order="C")
    if A.ndim != 2 or A.shape[1] != 2:
        A = np.array([[0.0, 0.0]], dtype="float64")
    mask = np.isfinite(A).all(axis=1)
    A = A[mask]
    if A.size == 0:
        return np.array([[0.0, 0.0]], dtype="float64")
    # enforce d>=b
    A[:, 1] = np.maximum(A[:, 1], A[:, 0])
    return np.ascontiguousarray(A, dtype="float64")


def pairwise_distances_safe(diags):
    """
    Compute pairwise distances without 
    multiprocessing; robust fallbacks.
    """
    
    n = len(diags)
    D = np.zeros((n, n), dtype="float64")
    for i in range(n):
        Xi = sanitize_diagram(diags[i])
        for j in range(i + 1, n):
            Xj = sanitize_diagram(diags[j])
            d = float(
                wasserstein_distance(
                    Xi, Xj, order=2.0, internal_p=2.0,
                    keep_essential_parts=False
                )
            )
            D[i, j] = D[j, i] = d
        if (i + 1) % 10 == 0:
            gc.collect()
    return D


def load_year_frames(pd_folder, Hidx, n_years):
    frames = []
    for idx in range(len(list(n_years))):
        path = os.path.join(pd_folder, f"PD_H_{Hidx}_{idx}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path)
        need = {"b", "d", "h_dim"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path} must have columns {need}, got {set(df.columns)}")
        frames.append(df)
    return frames


def pad_per_dim(frames, dims):
    by_dim_unpadded = {k: [] for k in dims}
    for df_year in frames:
        groups = dict(tuple(df_year.groupby("h_dim")))
        for k in dims:
            if k in groups:
                arr = groups[k][["b", "d"]].to_numpy(dtype=float)
                if arr.size == 0:
                    arr = np.array([[0.0, 0.0]], float)
            else:
                arr = np.array([[0.0, 0.0]], float)
            by_dim_unpadded[k].append(arr)
    pad = Padding()
    return {k: pad.fit_transform(v) for k, v in by_dim_unpadded.items()}


def compute_kernel_from_diagrams(diags, sigma):
    D = pairwise_distances_safe(diags)  # <— safe line
    K = np.exp(-(D ** 2) / (2.0 * (float(sigma) ** 2)))
    K = nearest_psd_correlation(K, min_eig=1e-8)
    return K


def precompute_kernels(pd_folder, cache_dir, n_years, dims, Hidx, sigmas, dtype_disk):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    frames = load_year_frames(pd_folder, Hidx, n_years)
    by_dim = pad_per_dim(frames, dims)

    meta = {
        "n_years": len(list(n_years)),
        "dims": list(dims),
        "Hidx": str(Hidx),
        "metric": "wasserstein_safe_loop",
        "dtype_disk": dtype_disk,
        "sigmas": {int(k): float((sigmas or {}).get(k, 1.0)) for k in dims},
        "version": "2-safe-loop",
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for d in dims:
        out_path = os.path.join(cache_dir, f"pd_kernel_h{d}.npz")
        if os.path.exists(out_path):
            print(f"skip H{d}: found {out_path}")
            continue
        print(f"compute H{d} (safe loop) ...")
        sigma = (sigmas or {}).get(d, 1.0)
        K = compute_kernel_from_diagrams(by_dim[d], sigma=sigma)
        np.savez_compressed(out_path, K=K.astype(dtype_disk, copy=False))
        print(f"saved {out_path}")

    print(f"✓ Precomputed kernels saved to: {cache_dir}")

def visualize_ecdf_all_dims_per_year(
    cache_dir,
    dims,
    selected_years,
    year_range,
    images_dir,
    xlim=(0.85, 1.001),
):
    """
    Visualize ECDFs of kernel similarity for multiple homology dimensions
    in 2×2 subplots (one per year), overlaid per dimension.
    """
    

    assert len(selected_years) == 4, "Exactly 4 years required for a 2×2 grid."

    start_year = year_range[0]
    year_indices = [y - start_year for y in selected_years]

    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    axs = axs.ravel()
    colors = sns.color_palette("tab10", len(dims))

    for i, (year, year_idx) in enumerate(zip(selected_years, year_indices)):
        ax = axs[i]
        for j, dim in enumerate(dims):
            kernel_path = os.path.join(cache_dir, f"pd_kernel_h{dim}.npz")
            if not os.path.exists(kernel_path):
                print(f"⚠️ Missing kernel: {kernel_path}")
                continue

            K = np.load(kernel_path)["K"]
            similarities = K[year_idx]

            sorted_vals = np.sort(similarities)
            ecdf = np.linspace(0, 1, len(sorted_vals))

            ax.plot(
                sorted_vals,
                ecdf,
                label=f"H{dim}",
                color=colors[j],
                linewidth=2,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Kernel Similarity")
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF: Similarity Distribution – Year {year}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Homology", fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.suptitle("ECDF of Kernel Similarities (H1–H4)", fontsize=16, y=1.02)
    output_path = os.path.join(images_dir, "pd_kernels_ecdf_H1_H4.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ ECDF plot saved to: {output_path}")


def visualize_all_similarity_distributions(
    cache_dir,
    dims,
    selected_years,
    year_range,
    images_dir,
    normalize=False,
):
    """
    Create boxplots, violin plots, and line plots of mean kernel similarities.
    
    Args:
        cache_dir (str): Path to directory with precomputed kernel matrices.
        dims (list[int]): Homology dimensions (e.g., [1,2,3,4]).
        selected_years (list[int]): Exactly 4 years to visualize per subplot.
        year_range (tuple[int, int]): (start_year, end_year)
        images_dir (str): Directory to save all plots.
        normalize (bool): If True, z-score normalize each kernel row before plotting.
    """
    

    start_year, end_year = year_range
    year_indices = [y - start_year for y in selected_years]
    all_years = list(range(start_year, end_year))
    all_year_indices = list(range(len(all_years)))
    
    # ---------- Collect Data ----------
    similarity_data = []
    mean_similarity = {f"H{d}": [] for d in dims}

    for d in dims:
        kernel_path = os.path.join(cache_dir, f"pd_kernel_h{d}.npz")
        if not os.path.exists(kernel_path):
            print(f"⚠️ Missing: {kernel_path}")
            continue

        K = np.load(kernel_path)["K"]
        
        if normalize:
            K = zscore(K, axis=1)

        # Collect row-wise values for selected years
        for i, y_idx in enumerate(year_indices):
            for val in K[y_idx]:
                similarity_data.append({
                    "Similarity": val,
                    "Homology": f"H{d}",
                    "Year": selected_years[i]
                })

        # Collect mean values for each year
        for i in all_year_indices:
            mean_similarity[f"H{d}"].append(np.mean(K[i]))

    df = pd.DataFrame(similarity_data)

    # ---------- 1. Boxplots ----------
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    axs = axs.ravel()
    for i, year in enumerate(selected_years):
        ax = axs[i]
        sns.boxplot(
            data=df[df["Year"] == year],
            x="Homology",
            y="Similarity",
            ax=ax,
            palette="tab10",
        )
        ax.set_title(f"Boxplot – Year {year}")
        ax.set_ylim(-3, 3 if normalize else 1.001)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.suptitle("Boxplots of Kernel Similarities (H1–H4)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(images_dir, "pd_kernels_boxplots.png"), dpi=300)
    plt.close()
    print("✓ Boxplots saved.")

    # ---------- 2. Violin Plots ----------
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    axs = axs.ravel()
    for i, year in enumerate(selected_years):
        ax = axs[i]
        sns.violinplot(
            data=df[df["Year"] == year],
            x="Homology",
            y="Similarity",
            ax=ax,
            palette="tab10",
            inner="box",
            cut=0,
        )
        ax.set_title(f"Violin Plot – Year {year}")
        ax.set_ylim(-3, 3 if normalize else 1.001)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.suptitle("Violin Plots of Kernel Similarities (H1–H4)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(images_dir, "pd_kernels_violins.png"), dpi=300)
    plt.close()
    print("✓ Violin plots saved.")

    # ---------- 3. Mean Similarity Over Time ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(dims))
    for i, (label, means) in enumerate(mean_similarity.items()):
        ax.plot(
            all_years,
            means,
            label=label,
            color=colors[i],
            linewidth=2,
        )
    ax.set_title("Mean Kernel Similarity Over Time", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Similarity (Z-score)" if normalize else "Mean Similarity")
    ax.set_ylim(-3, 3 if normalize else 1.001)
    ax.legend(title="Homology")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "pd_kernels_mean_over_time.png"), dpi=300)
    plt.close()
    print("✓ Mean similarity over time plot saved.")

def visualize_boxplots_by_homology_over_time(
    cache_dir,
    dims,
    year_range,
    images_dir,
    normalize=False,
):
    """
    Create 4 subplots (one per homology dimension) showing how kernel similarity
    distributions evolve over time via boxplots.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore

    start_year, end_year = year_range
    all_years = list(range(start_year, end_year))
    all_year_indices = list(range(len(all_years)))

    data = []

    for d in dims:
        kernel_path = os.path.join(cache_dir, f"pd_kernel_h{d}.npz")
        if not os.path.exists(kernel_path):
            print(f"⚠️ Missing kernel: {kernel_path}")
            continue

        K = np.load(kernel_path)["K"]
        if normalize:
            K = zscore(K, axis=1)

        for i, y in enumerate(all_years):
            for val in K[i]:
                data.append({
                    "Similarity": val,
                    "Year": y,
                    "Homology": f"H{d}"
                })

    df = pd.DataFrame(data)

    # ----- Plotting -----
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.ravel()

    for i, dim in enumerate(dims):
        ax = axs[i]
        sns.boxplot(
            data=df[df["Homology"] == f"H{dim}"],
            x="Year",
            y="Similarity",
            ax=ax,
            color=sns.color_palette("tab10")[i],
            fliersize=1,
        )
        ax.set_title(f"Kernel Similarity Over Time – H{dim}")
        ax.set_ylim(-3, 3 if normalize else 1.001)
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.suptitle("Temporal Evolution of PD Kernel Similarities by Homology Dimension", fontsize=16, y=1.02)
    out_path = os.path.join(images_dir, "pd_kernels_boxplot_by_H.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Boxplot over time saved: {out_path}")

def visualize_summary_stats_by_homology_over_time(
    cache_dir,
    dims,
    year_range,
    images_dir,
    normalize=False,
):
    """
    Plot mean, std, min, and max kernel similarity over time for each homology dimension.
    One subplot per Hk with shaded mean ± std and dashed min/max.
    """
   

    start_year, end_year = year_range
    all_years = list(range(start_year, end_year))
    year_indices = list(range(len(all_years)))

    # Store per-dimension stats
    stats_by_dim = {}

    for d in dims:
        kernel_path = os.path.join(cache_dir, f"pd_kernel_h{d}.npz")
        if not os.path.exists(kernel_path):
            print(f"⚠️ Missing kernel: {kernel_path}")
            continue

        K = np.load(kernel_path)["K"]
        if normalize:
            K = zscore(K, axis=1)

        means = np.mean(K, axis=1)
        stds = np.std(K, axis=1)
        mins = np.min(K, axis=1)
        maxs = np.max(K, axis=1)

        stats_by_dim[f"H{d}"] = {
            "mean": means,
            "std": stds,
            "min": mins,
            "max": maxs,
        }

    # ----- Plotting -----
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.ravel()

    for i, dim in enumerate(dims):
        label = f"H{dim}"
        stats = stats_by_dim.get(label)
        if stats is None:
            continue

        ax = axs[i]
        mean = stats["mean"]
        std = stats["std"]
        min_ = stats["min"]
        max_ = stats["max"]

        # Plot mean ± std
        ax.plot(all_years, mean, label="Mean", color="C0", linewidth=2)
        ax.fill_between(
            all_years,
            mean - std,
            mean + std,
            color="C0",
            alpha=0.2,
            label="±1 Std Dev",
        )

        # Plot min/max as dashed lines
        ax.plot(all_years, min_, label="Min", color="C1", linestyle="--", linewidth=1)
        ax.plot(all_years, max_, label="Max", color="C2", linestyle="--", linewidth=1)

        ax.set_title(f"Kernel Similarity – H{dim}")
        ax.set_ylabel("Similarity" + (" (z-score)" if normalize else ""))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

        # Reduce number of x-ticks for clarity
        if i >= 2:
            ax.set_xlabel("Year")
        ax.set_xticks(all_years[::10])
        ax.set_xticklabels(all_years[::10], rotation=45)

        if normalize:
            ax.set_ylim(-3, 3)
        else:
            ax.set_ylim(0.85, 1.001)

    plt.tight_layout()
    plt.suptitle("Summary of PD Kernel Similarity Over Time (H1–H4)", fontsize=16, y=1.02)
    out_path = os.path.join(images_dir, "pd_kernels_summary_by_H.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Summary plot saved: {out_path}")

def visualize_scatter_by_homology_over_time(
    cache_dir,
    dims,
    year_range,
    images_dir,
    normalize=False,
    show_means=True,
):
    """
    Scatterplot of similarity values over time for each homology dimension (H1–H4).
    Each subplot shows individual similarities as dots over years,
    with optional mean trend line.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore

    start_year, end_year = year_range
    all_years = list(range(start_year, end_year))

    data = []

    for d in dims:
        kernel_path = os.path.join(cache_dir, f"pd_kernel_h{d}.npz")
        if not os.path.exists(kernel_path):
            print(f"⚠️ Missing kernel: {kernel_path}")
            continue

        K = np.load(kernel_path)["K"]
        if normalize:
            K = zscore(K, axis=1)

        for i, y in enumerate(all_years):
            for val in K[i]:
                data.append({
                    "Similarity": val,
                    "Year": y,
                    "Homology": f"H{d}"
                })

    df = pd.DataFrame(data)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axs = axs.ravel()
    colors = sns.color_palette("tab10", len(dims))

    for i, dim in enumerate(dims):
        label = f"H{dim}"
        ax = axs[i]
        subset = df[df["Homology"] == label]

        # Main scatterplot
        sns.scatterplot(
            data=subset,
            x="Year",
            y="Similarity",
            ax=ax,
            color=colors[i],
            alpha=0.3,
            s=10,
            linewidth=0,
        )

        # Optional mean trend line
        if show_means:
            means = subset.groupby("Year")["Similarity"].mean()
            ax.plot(means.index, means.values, color=colors[i], linewidth=2, label="Mean")

        ax.set_title(f"Similarity Over Time – {label}")
        ax.set_ylabel("Similarity" + (" (z-score)" if normalize else ""))
        ax.set_xlabel("Year")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

        # Reduce clutter on x-axis
        ax.set_xticks(all_years[::10])
        ax.set_xticklabels(all_years[::10], rotation=45)

        if normalize:
            ax.set_ylim(-3, 3)
        else:
            ax.set_ylim(0.85, 1.001)

    plt.tight_layout()
    plt.suptitle("Kernel Similarity Over Time by Homology Dimension", fontsize=16, y=1.02)
    out_path = os.path.join(images_dir, "pd_kernels_scatter_by_H_fixed.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Fixed scatterplot saved: {out_path}")



if __name__ == "__main__":
    start, stop = CONFIG["years"]
    precompute_kernels(
        pd_folder=CONFIG["pd_folder"],
        cache_dir=CONFIG["cache_dir"],
        n_years=range(start, stop),
        dims=tuple(CONFIG["dims"]),
        Hidx=CONFIG["Hidx"],
        sigmas=dict(CONFIG["sigmas"]),
        dtype_disk=str(CONFIG["dtype_disk"]),
         )
    
    visualize_ecdf_all_dims_per_year(
        cache_dir=CONFIG["cache_dir"],
        dims=[1, 2, 3, 4],
        selected_years=[1955, 1975, 1995, 2010],
        year_range=CONFIG["years"],
        images_dir=CONFIG["images_dir"],
        xlim=(0.85, 1.001),  # adjust if needed
    )

    visualize_all_similarity_distributions(
        cache_dir=CONFIG["cache_dir"],
        dims=[1, 2, 3, 4],
        selected_years=[1955, 1975, 1995, 2010],
        year_range=CONFIG["years"],
        images_dir=CONFIG["images_dir"],
        normalize=True,  # or False for raw similarities
    )

    visualize_boxplots_by_homology_over_time(
        cache_dir=CONFIG["cache_dir"],
        dims=[1, 2, 3, 4],
        year_range=CONFIG["years"],
        images_dir=CONFIG["images_dir"],
        normalize=True,
    )

    visualize_summary_stats_by_homology_over_time(
        cache_dir=CONFIG["cache_dir"],
        dims=[1, 2, 3, 4],
        year_range=CONFIG["years"],
        images_dir=CONFIG["images_dir"],
        normalize=True,  # or False for raw similarity
    )

    visualize_scatter_by_homology_over_time(
        cache_dir=CONFIG["cache_dir"],
        dims=[1, 2, 3, 4],
        year_range=CONFIG["years"],
        images_dir=CONFIG["images_dir"],
        normalize=True,
        show_means=True,
    )
