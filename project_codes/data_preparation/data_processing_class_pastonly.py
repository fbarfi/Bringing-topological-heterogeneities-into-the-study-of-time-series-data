
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyreadstat


class PaxPetrollicaDataProcessor:
    """
    Builds the rectangular 182-state x 66-year (1945-2010) panel from
    PaxPetrolica_Main.dta and fills its missing cells.

      * object columns (revleader, new_oil_state, polity, borders) are parsed to
        numeric, so their missing values appear as NaN;
      * within-history gaps (a state exists in year t but a covariate is
        missing) are filled by within-country forward-fill; gaps before a
        state's first observed value are filled by within-country back-fill
        from that first observed value;
      * pre-existence cells (years before a late-entry state appears in the
        source) are filled from the same-year cross-sectional median over the
        states that exist that year. These cells exist because the panel is
        rectangularized to the full 182 x 66 system grid, which gives a
        comparable year-by-year point cloud for the topological stage;
      * cells still missing after these steps (covariates never observed for a
        given state) are filled from the median of that covariate over the
        whole panel.

    Outputs are written under *_augm names (the names the downstream loaders
    read); set OUT_SUFFIX to change.
    """

    OUT_SUFFIX = '_augm'   # set to e.g. '_augm_pastonly' to avoid overwriting

    def __init__(self):
        self.data_path = "../../project_data/PaxPetrolica_Main.dta"
        self.images = "../../project_data/images/"
        os.makedirs(self.images, exist_ok=True)

        # object columns that must be parsed to numeric (NaN currently hidden)
        self.object_cols = ['revleader', 'new_oil_state', 'polity', 'borders']
        # variables stored as integers in the output
        self.int_vars = ['revleader', 'new_oil_state', 'coldwar', 'majorpower',
                         'polity', 'borders']
        self.float_vars = ['logoil', 'ln_gdppc', 'lnpop', 'muslim']
        self.analysis_cols = self.float_vars + self.int_vars

        # known true missing counts (for the parse-integrity assertion)
        self.expected_missing = {
            'revleader': 566, 'new_oil_state': 955, 'logoil': 1028,
            'ln_gdppc': 209, 'lnpop': 139, 'muslim': 274,
            'polity': 912, 'borders': 156,
        }

    # ------------------------------------------------------------------ load
    def load_data(self):
        self.in_data, self.meta = pyreadstat.read_dta(self.data_path)

    def save_variable_labels(self):
        pd.concat([
            pd.DataFrame({'variable': self.meta.column_names}),
            pd.DataFrame({'label': self.meta.column_labels})
        ], axis=1).to_csv('../../project_data/variables_labels.csv')

    # ------------------------------------------------------- parse + select
    def preprocess_data(self):
        df = self.in_data[['stabb', 'ccode', 'ccname', 'year', 'revonsets', 'logoil',
                           'revolutionaryleader', 'new_oil_state', 'ln_gdppc', 'lnpop',
                           'polity', 'borders', 'coldwar', 'muslim', 'majorpower']].copy()
        df['ccode'] = df['ccode'].replace({818: 816})
        df.rename(columns={'revolutionaryleader': 'revleader'}, inplace=True)

        # Parse object columns to numeric. errors='coerce' turns any genuine
        # non-numeric token into NaN; the assertion below guarantees we only
        # surface the KNOWN missing values and never silently destroy a valid
        # number.
        for c in self.object_cols:
            before_nan = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors='coerce')
            after_nan = df[c].isna().sum()
            created = int(after_nan - before_nan)
            expected = self.expected_missing.get(c, None)
            # parsing should expose exactly the known missing count (no more)
            if expected is not None and int(after_nan) != expected:
                raise ValueError(
                    f"Parse integrity check failed for '{c}': "
                    f"{int(after_nan)} NaN after parse, expected {expected}. "
                    f"({created} new NaN created by coercion -- a non-numeric "
                    f"token may have destroyed a valid value.)")

        self.df_data = df.reset_index(drop=True)
        print("Parsed object columns to numeric; missingness now honest:")
        print(self.df_data[self.analysis_cols].isna().sum().to_string())

    def save_states(self):
        states = self.df_data[['stabb', 'ccode', 'ccname']].copy()
        states.drop_duplicates('ccode', inplace=True)
        states.to_csv('../../project_data/states_names_ccodes.csv')

    # ----------------------------------------------- build full system grid
    def build_full_grid(self):
        """
        Rectangularize to the full 182 x 66 system grid. Cells for country-years
        present in the source keep their (possibly missing) values; cells for
        country-years ABSENT from the source (pre-existence years of late-entry
        states) are introduced as NaN and tagged so they can be filled
        contemporaneously rather than by forward-fill.
        """
        self.year_list = sorted(self.df_data['year'].unique())
        self.all_ccodes = np.unique(self.df_data['ccode'].values)
        n_y = len(self.year_list)

        full_index = pd.MultiIndex.from_product(
            [self.all_ccodes, self.year_list], names=['ccode', 'year'])

        keep = ['ccode', 'year', 'revonsets'] + self.analysis_cols
        src = self.df_data[keep].drop_duplicates(['ccode', 'year']) \
                                .set_index(['ccode', 'year'])

        grid = src.reindex(full_index)

        # a row is "pre-existence scaffolding" if that (ccode, year) was absent
        # from the source entirely (all source columns NaN because the row did
        # not exist). Detect via the target revonsets, which has no real NaN in
        # the source.
        existed = src.index
        grid['_exists'] = grid.index.isin(existed)
        self.grid = grid.reset_index()

        n_real = int(self.grid['_exists'].sum())
        n_scaffold = int((~self.grid['_exists']).sum())
        print(f"System grid: {len(self.all_ccodes)} states x {n_y} years "
              f"= {len(self.grid)} cells "
              f"({n_real} real country-years, {n_scaffold} pre-existence "
              f"scaffolding cells).")

    # ----------------------------------------------------- past-only impute
    def impute_pastonly(self):
        """
        Imputation in three steps:

          1. WITHIN-HISTORY gaps (a state exists in year t but a covariate is
             missing): within-country forward-fill from that state's earlier
             observations. Gaps before a state's first observed value are then
             filled by within-country back-fill from that first observed value,
             restricted to years in which the state exists.

          2. PRE-EXISTENCE scaffolding cells (state did not exist in year t):
             filled from the same-year cross-sectional median over states that
             exist that year.

          3. Cells still missing (a covariate never observed for a state) are
             filled from the median of that covariate over the whole panel.
        """
        g = self.grid.sort_values(['ccode', 'year']).copy()
        cols = self.analysis_cols + ['revonsets']
        exists = g['_exists'].values

        # --- stage 1: within-country fills, restricted to existing years -----
        # operate only on rows where the state exists, so scaffolding cells do
        # not absorb a forward-filled value from a later real year (which would
        # be a future value flowing backward across the existence boundary).
        g_real = g[g['_exists']].copy()
        g_real[cols] = g_real.groupby('ccode')[cols].ffill()
        g_real[cols] = g_real.groupby('ccode')[cols].bfill()  # leading within-history gaps
        # write the filled real rows back
        g.loc[g['_exists'].values, cols] = g_real[cols].values

        # --- stage 2: contemporaneous fill for pre-existence scaffolding ------
        for c in cols:
            same_year = g.groupby('year')[c].transform('median')
            fill_mask = (~g['_exists']) & (g[c].isna())
            g.loc[fill_mask, c] = same_year[fill_mask]

        # final safety net (a column with no real value in some year at all)
        residual = int(g[cols].isna().sum().sum())
        if residual > 0:
            print(f"  note: {residual} residual cells filled from global median "
                  f"(contemporaneous fallback exhausted).")
            g[cols] = g[cols].fillna(g[cols].median())

        # integer casts for the categorical / count variables
        for v in self.int_vars:
            g[v] = g[v].round().astype(int)
        g['revonsets'] = g['revonsets'].round().astype(int)

        # output column order:
        # year, ccode, revonsets, revleader, new_oil_state, coldwar, majorpower,
        # logoil, ln_gdppc, lnpop, muslim, polity, borders
        ordered = ['year', 'ccode', 'revonsets', 'revleader', 'new_oil_state',
                   'coldwar', 'majorpower', 'logoil', 'ln_gdppc', 'lnpop',
                   'muslim', 'polity', 'borders']
        self.df_augm = g.sort_values(['ccode', 'year'])[ordered].reset_index(drop=True)

        # verify balanced + complete
        counts = self.df_augm.groupby('ccode')['year'].nunique()
        n_c, n_y = len(self.all_ccodes), len(self.year_list)
        balanced = (counts == n_y).all() and len(self.df_augm) == n_c * n_y
        print(f"Imputation complete: {len(self.df_augm)} rows, balanced={balanced}, "
              f"NaN remaining={bool(self.df_augm.isna().any().any())}")

    # ------------------------------------------------------------- outputs
    def save_histogram(self):
        d = self.df_augm.loc[self.df_augm['revonsets'] > 0, self.analysis_cols]
        d.plot(kind='hist', bins=50, subplots=True, layout=(5, 3),
               figsize=(15, 20), sharex=False, sharey=False)
        plt.savefig(self.images + 'hist_all_vars' + self.OUT_SUFFIX + '.png')
        plt.close()

    def save_final_datasets(self):
        s = self.OUT_SUFFIX
        self.df_augm.to_csv(f'../../project_data/dataset{s}.csv')
        self.df_augm[['year', 'ccode', 'revonsets']].to_hdf(
            f'../../project_data/dv{s}.h5', key='data', mode='w', complevel=9)
        self.df_augm[['year', 'ccode', 'revleader', 'logoil', 'new_oil_state',
                      'ln_gdppc', 'lnpop', 'polity', 'borders', 'coldwar',
                      'muslim', 'majorpower']].to_hdf(
            f'../../project_data/idvs{s}.h5', key='data', mode='w', complevel=9)

    def process_data(self):
        self.load_data()
        self.save_variable_labels()
        self.preprocess_data()
        self.save_states()
        self.build_full_grid()
        self.impute_pastonly()
        self.save_histogram()
        self.save_final_datasets()
        print("Data processing (past-only / contemporaneous imputation) "
              "completed successfully.")


if __name__ == "__main__":
    processor = PaxPetrollicaDataProcessor()
    processor.process_data()
