# project_data

The data used in this study are not redistributed in this repository.

## Download

1. Go to the Harvard Dataverse record for Jang and Smith (2021):
   <https://doi.org/10.7910/DVN/V1VESK>
2. Download `PaxPetrolica_Main.dta`.
3. Put it in this folder:

```
project_data/PaxPetrolica_Main.dta
```

## Files written by the pipeline

The `data` stage (`project_codes/data_preparation/data_processing_class_pastonly.py`)
reads `PaxPetrolica_Main.dta` and writes:

| File | Content |
|---|---|
| `dataset_augm.csv` | balanced 182 states x 66 years (1945–2010) panel: `year, ccode, revonsets` and the ten covariates |
| `dv_augm.h5` (key `data`) | `year, ccode, revonsets` |
| `idvs_augm.h5` (key `data`) | `year, ccode` and the ten covariates |
| `states_names_ccodes.csv` | `stabb, ccode, ccname` for every state |
| `variables_labels.csv` | variable names and Stata labels |
| `images/hist_all_vars_augm.png` | covariate histograms (country-years with `revonsets > 0`) |
| `dist_check_out/` | observed-vs-imputed distribution checks (`distribution_check_pastonly.py`) |

Covariates: `revleader, new_oil_state, coldwar, majorpower, logoil, ln_gdppc,
lnpop, muslim, polity, borders`. Country code 818 is recoded to 816.
