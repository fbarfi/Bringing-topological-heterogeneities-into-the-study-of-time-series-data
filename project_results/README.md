# project_results

All pipeline outputs are written here. The folder is empty in the repository.

| Folder | Written by | Content |
|---|---|---|
| `gudhi_multiple_tda_results_H_4/` | `persistence_diagrams/` | embedded point clouds, persistence diagrams `PersistenceDiagrams/PD_H_1234_<t>.csv` (one per year, columns `b, d, h_dim`), functional summaries, Betti curves, PD-kernel caches, figures |
| `tsai_no_tda_Results/classification_models/epochs_900/` | `bare_classifier/` | per backbone and fold: trained learner, history, predictions (`fold_predictions.npz`), feature importances; `cv_final/` figures per backbone |
| `tsai_yes_tda_gpd_Results/classification_models/epochs_900/` | `tda_classifier/` | same layout for the TDA-conditioned classifier, plus PDDeepSets / prior diagnostics |
| `tsai_comparing_models_gpd_results/nepochs_900/` | `model_comparison/` | paired comparison outputs: `test_cv/` (per-backbone comparisons, summary plots, calibration, tables, pooled Wilcoxon tests), `cv_pooled/`, `beta_attention/` |
