"""
distribution_check_pastonly.py

Compares the imputed panel (data_processing_class_pastonly.py output) against
the observed values in PaxPetrolica_Main.dta, per variable.

Three cell categories are separated, because the rectangularized 182x66 panel
contains cells of different kinds:

  OBSERVED      : present and non-missing in the source .dta.
  IMPUTED       : a country-year that EXISTED in the source but had a missing
                  covariate -> filled by within-country forward-fill. This is
                  the real test of imputation faithfulness.
  SCAFFOLDING   : a country-year that did NOT exist in the source (pre-existence
                  years of late-entry states) -> filled contemporaneously. A
                  different question (panel construction, not imputation), so it
                  is reported separately and EXCLUDED from the imputation-quality
                  comparison.

For each continuous variable it reports, on the genuinely-IMPUTED cells only:
mean/sd/min/max and a Kolmogorov-Smirnov statistic vs the OBSERVED cells. It
also reports the scaffolding-cell distribution separately for transparency.

Outputs CSV + LaTeX + overlay histograms.

Run:
    python distribution_check_pastonly.py
Set paths in CONFIG.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
RAW_DTA = "../../project_data/PaxPetrolica_Main.dta"
IMPUTED_CSV = "../../project_data/dataset_augm.csv"   # pastonly processor output
IMAGES_DIR = "../../project_data/images/"
OUT_DIR = "../../project_data/dist_check_out/"

FLOAT_VARS = ['logoil', 'ln_gdppc', 'lnpop', 'muslim']
INT_VARS = ['revleader', 'new_oil_state', 'polity', 'borders', 'coldwar', 'majorpower']
ALL_VARS = FLOAT_VARS + INT_VARS

# object columns in the raw .dta that must be parsed to numeric to expose NaN
OBJECT_COLS = ['revleader', 'new_oil_state', 'polity', 'borders']

# polity and borders are continuous per the author -> KS, not category shares
CONTINUOUS_FOR_KS = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 'polity', 'borders']
CATEGORICAL = ['revleader', 'new_oil_state', 'coldwar', 'majorpower']


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
def load_original():
    import pyreadstat
    df, _ = pyreadstat.read_dta(RAW_DTA)
    df = df.copy()
    df['ccode'] = df['ccode'].replace({818: 816})
    df = df.rename(columns={'revolutionaryleader': 'revleader'})
    # parse object columns to numeric the SAME way the processor does, so the
    # true missingness is exposed instead of hidden as a category code.
    for c in OBJECT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    keep = ['ccode', 'year'] + [c for c in ALL_VARS if c in df.columns]
    return df[keep]


def load_imputed():
    df = pd.read_csv(IMPUTED_CSV)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


# ----------------------------------------------------------------------------
# Three-way cell classification.
#   - a (ccode, year) is REAL if it appears in the source at all
#   - within a real cell, a variable is OBSERVED if non-NaN, else IMPUTED
#   - a (ccode, year) absent from source is SCAFFOLDING for every variable
# ----------------------------------------------------------------------------
def classify_cells(orig, imp):
    orig_idx = orig.set_index(['ccode', 'year'])
    imp_idx = imp.set_index(['ccode', 'year'])

    real_keys = set(map(tuple, orig[['ccode', 'year']].drop_duplicates().values))
    is_real = pd.Series([k in real_keys for k in imp_idx.index], index=imp_idx.index)

    obs_mask = pd.DataFrame(False, index=imp_idx.index, columns=ALL_VARS)
    imp_mask = pd.DataFrame(False, index=imp_idx.index, columns=ALL_VARS)
    scaf_mask = pd.DataFrame(False, index=imp_idx.index, columns=ALL_VARS)

    for v in ALL_VARS:
        if v not in orig_idx.columns:
            scaf_mask[v] = True
            continue
        orig_v = orig_idx[v].reindex(imp_idx.index)
        scaffolding = ~is_real
        observed = is_real & orig_v.notna().values
        imputed = is_real & orig_v.isna().values
        obs_mask[v] = observed.values
        imp_mask[v] = imputed.values
        scaf_mask[v] = scaffolding.values

    return orig_idx, imp_idx, obs_mask, imp_mask, scaf_mask


def desc(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, min=np.nan, max=np.nan)
    return dict(n=len(x), mean=x.mean(), sd=x.std(), min=x.min(), max=x.max())


def ks(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if not HAVE_SCIPY or len(a) == 0 or len(b) == 0:
        return np.nan
    return ks_2samp(a, b).statistic


def report(orig_idx, imp_idx, obs_mask, imp_mask, scaf_mask):
    print("=" * 82)
    print("DISTRIBUTION COMPARISON: original observed vs past-only imputed")
    print("=" * 82)
    print("""
Cell categories per variable:
  OBSERVED    = present & non-missing in source
  IMPUTED     = existing country-year, missing covariate -> forward-filled
  SCAFFOLDING = pre-existence country-year -> contemporaneous fill (reported
                separately; NOT part of the imputation-quality KS)
KS compares OBSERVED vs IMPUTED only (0 = identical, 1 = disjoint).
""")
    summary, cat_rows = [], []

    for v in ALL_VARS:
        print("-" * 82)
        print(f"VARIABLE: {v}")
        if v not in orig_idx.columns:
            print("  (absent in source)")
            summary.append(dict(variable=v, kind='absent'))
            continue

        observed = imp_idx[v].values[obs_mask[v].values]
        imputed = imp_idx[v].values[imp_mask[v].values]
        scaffold = imp_idx[v].values[scaf_mask[v].values]

        n_obs, n_imp, n_scaf = len(observed), len(imputed), len(scaffold)
        print(f"  observed={n_obs}  imputed={n_imp}  scaffolding={n_scaf}")

        if v in CONTINUOUS_FOR_KS:
            ks_oi = ks(observed, imputed)
            do, di, ds = desc(observed), desc(imputed), desc(scaffold)
            print(f"  KS(observed vs imputed) = {ks_oi:.4f}")
            print(f"    observed    : mean={do['mean']:8.3f} sd={do['sd']:7.3f}")
            print(f"    imputed     : mean={di['mean']:8.3f} sd={di['sd']:7.3f}")
            print(f"    scaffolding : mean={ds['mean']:8.3f} sd={ds['sd']:7.3f} "
                  f"(separate; not in KS)")
            summary.append(dict(
                variable=v, kind='continuous',
                n_observed=n_obs, n_imputed=n_imp, n_scaffolding=n_scaf,
                ks_obs_vs_imp=None if np.isnan(ks_oi) else round(ks_oi, 4),
                obs_mean=round(do['mean'], 4), obs_sd=round(do['sd'], 4),
                imp_mean=round(di['mean'], 4), imp_sd=round(di['sd'], 4),
                scaf_mean=round(ds['mean'], 4), scaf_sd=round(ds['sd'], 4),
            ))
        else:
            def shares(x):
                x = pd.Series(x).dropna().astype(int)
                return x.value_counts(normalize=True).sort_index()
            s_o, s_i = shares(observed), shares(imputed)
            cats = sorted(set(s_o.index) | set(s_i.index))
            print("    category shares  observed | imputed")
            for cat in cats:
                print(f"      {cat:>3}: {s_o.get(cat,0):.3f} | {s_i.get(cat,0):.3f}")
                cat_rows.append(dict(variable=v, category=int(cat),
                                     share_observed=round(float(s_o.get(cat, 0)), 4),
                                     share_imputed=round(float(s_i.get(cat, 0)), 4)))
            tv = 0.5 * sum(abs(s_o.get(c, 0) - s_i.get(c, 0)) for c in cats)
            print(f"    total-variation(observed vs imputed) = {tv:.4f}")
            summary.append(dict(variable=v, kind='categorical',
                                n_observed=n_obs, n_imputed=n_imp,
                                n_scaffolding=n_scaf,
                                tv_obs_vs_imp=round(tv, 4)))

    return pd.DataFrame(summary), pd.DataFrame(cat_rows)


def export(summary, cat):
    os.makedirs(OUT_DIR, exist_ok=True)
    sp = os.path.join(OUT_DIR, "dist_check_pastonly_summary.csv")
    summary.to_csv(sp, index=False); print(f"saved {sp}")
    if len(cat) > 0:
        cp = os.path.join(OUT_DIR, "dist_check_pastonly_categorical.csv")
        cat.to_csv(cp, index=False); print(f"saved {cp}")

    # LaTeX
    def esc(s): return str(s).replace('_', r'\_')
    cont = summary[summary['kind'] == 'continuous']
    catg = summary[summary['kind'] == 'categorical']
    L = [r"% Auto-generated by distribution_check_pastonly.py",
         r"\begin{table}[htbp]", r"\centering", r"\small",
         r"\caption{Distributional comparison of past-only imputed covariates "
         r"against the original observed data. KS compares observed cells to "
         r"genuinely-imputed cells (existing country-years with missing values, "
         r"forward-filled). Pre-existence scaffolding cells are excluded. Larger "
         r"KS = greater distributional difference.}",
         r"\label{tab:dist_check_pastonly}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Variable & $n_{\mathrm{obs}}$ & $n_{\mathrm{imp}}$ & KS & "
         r"obs mean (sd) & imp mean (sd) \\", r"\midrule"]
    for _, r in cont.iterrows():
        zero_imp = int(r['n_imputed']) == 0
        ks_str = "--" if r['ks_obs_vs_imp'] is None or pd.isna(r['ks_obs_vs_imp']) \
            else f"{r['ks_obs_vs_imp']:.4f}"
        imp_str = "--" if zero_imp else f"{r['imp_mean']:.3f} ({r['imp_sd']:.3f})"
        L.append(f"{esc(r['variable'])} & {int(r['n_observed'])} & "
                 f"{int(r['n_imputed'])} & {ks_str} & "
                 f"{r['obs_mean']:.3f} ({r['obs_sd']:.3f}) & {imp_str} \\\\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{6}{l}{\textit{Categorical (total-variation, "
             r"observed vs.\ imputed):}} \\")
    for _, r in catg.iterrows():
        tv_str = "-- (no imputed cells)" if int(r['n_imputed']) == 0 \
            else f"TV = {r['tv_obs_vs_imp']:.4f}"
        L.append(f"{esc(r['variable'])} & {int(r['n_observed'])} & "
                 f"{int(r['n_imputed'])} & "
                 f"\\multicolumn{{3}}{{l}}{{{tv_str}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tp = os.path.join(OUT_DIR, "dist_check_pastonly_summary.tex")
    with open(tp, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"saved {tp}")


def plot_overlays(imp_idx, obs_mask, imp_mask):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    n = len(ALL_VARS); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, v in enumerate(ALL_VARS):
        ax = axes[i]
        observed = imp_idx[v].values[obs_mask[v].values]
        imputed = imp_idx[v].values[imp_mask[v].values]
        ax.hist(observed, bins=40, density=True, alpha=0.5, label='observed')
        if len(imputed) > 0:
            ax.hist(imputed, bins=40, density=True, alpha=0.5, label='imputed')
        ax.set_title(v); ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, 'dist_observed_vs_imputed_pastonly.png')
    fig.savefig(out, dpi=110); plt.close()
    print(f"saved overlay -> {out}")


def main():
    orig = load_original()
    imp = load_imputed()
    orig_idx, imp_idx, obs_mask, imp_mask, scaf_mask = classify_cells(orig, imp)
    summary, cat = report(orig_idx, imp_idx, obs_mask, imp_mask, scaf_mask)
    export(summary, cat)
    try:
        plot_overlays(imp_idx, obs_mask, imp_mask)
    except Exception as e:
        print(f"(overlay skipped: {e})")
    print("\nDone.")


if __name__ == "__main__":
    main()
