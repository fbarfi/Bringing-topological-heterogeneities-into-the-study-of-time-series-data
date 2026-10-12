# Bringing topological heterogeneities into the study of time-series data

Replication code for:

> Arfi, B. (2026). Bringing topological heterogeneities into the study of time-series data on militarized international disputes: a deep learning approach. *Journal of Computational Social Science*, 9, 67. <https://doi.org/10.1007/s42001-026-00499-8>

The code extracts persistence diagrams (PDs) from the year-by-year configuration of the interstate system and uses them to condition a supervised deep classifier of country covariate sequences. A TDA-conditioned classifier and a bare (no-TDA) classifier are trained on ten [tsai](https://github.com/timeseriesAI/tsai) backbones under the same 5-fold, country-level cross-validation. The two are then compared on paired folds.

## Pipeline

```
PaxPetrolica_Main.dta
   │
   ├─ 1. data_preparation       182 states × 66 years (1945–2010) panel
   │
   ├─ 2. persistence_diagrams   state-axis delay lift (D = 30, τ = 1) → 182 × 300 point cloud per year
   │                            → Vietoris–Rips persistence H0–H4 → one PD per year
   │
   ├─ 3. bare_classifier        tsai backbone + classification head          (10 backbones × 5 folds)
   ├─ 4. tda_classifier         tsai backbone + PD encoder + point-level      (10 backbones × 5 folds)
   │                            cross-attention conditioner + BiGRU fusion
   │                            + VAE with Product-of-Experts topological prior
   │
   └─ 5. model_comparison       paired bare-vs-TDA metrics, calibration, feature-importance ranks,
                                LaTeX tables, pooled Wilcoxon tests, attention diagnostics
```

## Repository layout

```
run_pipeline.py                  runs the stages in order
requirements.txt / environment.yml
project_data/                    input data (downloaded separately; see project_data/README.md)
project_results/                 all outputs (created by the pipeline)
project_codes/
    data_preparation/
        data_processing_class_pastonly.py     builds and imputes the balanced panel
        distribution_check_pastonly.py        observed-vs-imputed distribution checks
    persistence_diagrams/
        run_all_gudhi_scripts.py              runs the two scripts below
        main_gudhi_persistence.py             delay lift, persistence diagrams, functional summaries
        main_plotting_gudhi_representations.py  PD / barcode / summary / Betti figures
        tda_precompute_kernels.py             PD-kernel similarities and figures
        gudhi_python_codes/
    bare_classifier/
        main_classification.py                trains and cross-validates the bare models
        finalize_all.py                       CV figures and metrics per backbone
        fold_class_support.py                 per-fold class counts
        classification_python_codes/
    tda_classifier/
        main.py                               entry point (stages: classify, finalize_all,
                                              finalize_one, pd_deepsets, fold_support)
        main_classification.py                trains and cross-validates the TDA models
        finalize_all.py, fold_class_support.py, run_pd_deepsets_all_backbones.py
        classification_python_codes/
            tsai_classifier_VAE_class.py      model definition
            models.py                         training loop, composite loss, callbacks
            preparing_data_class.py           PD-aware datasets and fold construction
    model_comparison/
        compare_all.py                        paired comparison (subcommands below)
        aggregate_beta_attention.py           fold-aggregated conditioning-attention diagnostics
        comparing_codes/
```

## Installation

Python 3.10–3.12. With conda:

```bash
conda env create -f environment.yml
conda activate tda-mid
```

or with pip:

```bash
pip install -r requirements.txt
```

The device is detected automatically, in this order: Apple-silicon MPS, CUDA, CPU.

## Data

The data are those of Jang and Smith (2021), available on Harvard Dataverse at <https://doi.org/10.7910/DVN/V1VESK>. Download `PaxPetrolica_Main.dta` and put it in `project_data/` (see [`project_data/README.md`](project_data/README.md)).

## Running

The complete pipeline:

```bash
python run_pipeline.py
```

One or more stages:

```bash
python run_pipeline.py --list            # stages and their steps
python run_pipeline.py data tda
python run_pipeline.py bare tda_model
python run_pipeline.py compare
```

Each script can also be run directly from its own directory. All paths are relative to that directory (`../../project_data/`, `../../project_results/`):

```bash
cd project_codes/tda_classifier
python main.py --list
python main.py classify finalize_all pd_deepsets

cd ../model_comparison
python compare_all.py all        # engine (per-fold comparison, plots, importance, calibration) + pooled
python compare_all.py tables     # LaTeX performance and overfitting tables
python compare_all.py stats      # pooled Wilcoxon signed-rank tests
python aggregate_beta_attention.py --device cpu
```

The `bare` and `tda_model` stages train 10 backbones × 5 folds, for up to 900 epochs each with early stopping. Training is the expensive part of the pipeline. The comparison stage reads the trained learners (`trained_classification_model.pkl`) and per-fold predictions written by the two training stages.

## Main settings

| Setting | Value |
|---|---|
| Units / label | 182 countries; label = maximum collapsed `revonsets` category over 1945–2010 (values > 2 → class 3; 4 classes) |
| Cross-validation | stratified 5-fold over countries, seed 42; validation = 15% of each training pool (stratified, seed 42); identical folds for both classifiers |
| Covariate scaling | per-country `StandardScaler` on `logoil, ln_gdppc, lnpop, muslim, borders, polity`; binary covariates unscaled |
| Point cloud per year | continuous covariates min–max scaled over the panel, binary covariates unscaled; state-axis delay lift with D = 30, τ = 1 over the ccode ordering (front padding keeps all 182 states) → 182 × 300 |
| Persistence | Vietoris–Rips, H0–H4; finite points; birth and death values min–max scaled on one common scale over H0–H4 of each year; H1–H4 enter the classifier |
| Backbones | RNNPlus, TSiT, TSiTPlus, ResNetPlus, InceptionTimePlus, XCMPlus, TSSequencerPlus, RNNAttention, LSTMAttention, GRUAttention |
| TDA classifier | hidden 128, latent 128, PD encoder width 128, 2-layer BiGRU, batch 8; focal loss with inverse-frequency class weights + β·(reconstruction + KL to PoE prior) + λ_cond·conditioner-auxiliary loss |
| Optimization (TDA) | Adam (0.9, 0.98), one-cycle, lr_max 1e-3, wd 1e-2, gradient clip 0.1, ≤ 900 epochs, early stopping on validation loss (patience 20, min_delta 0.005); λ_cond updated each epoch by dual ascent on the validation loss with vs. without the conditioner |
| Optimization (bare) | Adam, one-cycle, lr_max 1e-5, wd 1e-5, ≤ 900 epochs, focal loss with inverse-frequency class weights; early stopping, best-checkpoint restore and ReduceLROnPlateau on validation loss (patience 20, min_delta 1e-5); parameter EMA (0.99) used for validation |

## Where each result is produced

Paths are under `project_results/`.

| Article | Script | Output |
|---|---|---|
| Fig. 2 (PDs, four years) | `persistence_diagrams/main_plotting_gudhi_representations.py` | `gudhi_multiple_tda_results_H_4/images/` |
| Fig. 3 (PD-kernel similarities) | `persistence_diagrams/tda_precompute_kernels.py` | `gudhi_multiple_tda_results_H_4/images/pd_kernels_boxplots.png` |
| Figs. 4–5 (conditioning attention by homology dimension; concentration) | `model_comparison/aggregate_beta_attention.py` | `tsai_comparing_models_gpd_results/nepochs_900/beta_attention/` |
| Figs. 6–7 (per-year prior and PoE prior) | `tda_classifier/main.py pd_deepsets` | `tsai_comparing_models_gpd_results/nepochs_900/test_cv/summary_plots/pd_deepsets/<backbone>/` |
| Table 1 (TDA performance) | `model_comparison/compare_all.py tables` | `…/test_cv/tables/` |
| Figs. 10–11 (ROC, RNNAttention) | `tda_classifier/finalize_all.py` | `tsai_yes_tda_gpd_Results/…/RNNAttention_val_0.2_test_0.1/cv_final/` |
| Table 2 (per-fold class counts) | `tda_classifier/fold_class_support.py` | `tsai_yes_tda_gpd_Results/fold_support_out/` |
| Figs. 12–17 (importance ranks) | `model_comparison/compare_all.py all` (engine → importance) | `…/test_cv/importance_rank_<metric>_<split>_tda_vs_notda.png` |
| Table 3 (train/test accuracy, base vs TDA) | `model_comparison/compare_all.py tables` | `…/test_cv/tables/` |
| Table 4 (pooled Wilcoxon tests) | `model_comparison/compare_all.py stats` | `…/test_cv/tables/pooled_wilcoxon_results.{csv,tex}` |
| Figs. 18–23 (confidence, reliability, mean metrics, deltas, wins, ΔAccuracy) | `model_comparison/compare_all.py all` (engine → plots) | `…/test_cv/summary_plots/` |
| Tables 5–6 (calibration modes) | `model_comparison/compare_all.py all` (engine → calibration) | `…/test_cv/calibration/` |

## Reproducibility

Seeds (`random`, `numpy`, `torch`) are fixed per fold (42 + fold index), and the fold indices are fixed (seed 42). GPU and MPS kernels are not fully deterministic, so retrained models can differ slightly from the published ones across hardware and library versions.

## Citation

```bibtex
@article{Arfi2026topological,
  author  = {Arfi, Badredine},
  title   = {Bringing topological heterogeneities into the study of time-series data on militarized international disputes: a deep learning approach},
  journal = {Journal of Computational Social Science},
  year    = {2026},
  volume  = {9},
  pages   = {67},
  doi     = {10.1007/s42001-026-00499-8}
}
```

Please also cite the data source: Jang, H. R., & Smith, B. (2021). Pax Petrolica? Rethinking the oil–interstate war linkage. *Security Studies*, 30(2), 159–181.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The data are not part of this repository and remain under the terms of their source.

## Contact

Badredine Arfi, Department of Political Science, University of Florida — barfi@ufl.edu
