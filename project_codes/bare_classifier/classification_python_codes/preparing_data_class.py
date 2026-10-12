import os
from tsai.all import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
Label_Encoder = LabelEncoder()
from sklearn.model_selection import train_test_split, StratifiedKFold
from fastai.data.core import DataLoaders
import torch

from .base_dataloader import TSDataLoaderClass

class PreparingDataClass:
    def __init__(self, args):
        super(PreparingDataClass, self).__init__()

        self.args = args
        self.num_cat = 5
        self.n_years = args['n_years']
        self.float_var = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity']
        self.category_var = ['revleader', 'new_oil_state', 'coldwar', 'majorpower']
        self.feature_names = args['features']
        self.val_size = args['val_size']
        self.test_size = args['test_size']
        self.bs = args['bs']
        self.scaler = StandardScaler() # MinMaxScaler()

    def loading_data(self):

        # Load dependent variable dataset
        df_dv = pd.read_hdf('../../project_data/dv_augm.h5', key='data').sort_values(by=['ccode', 'year'])

        # Categorize revonsets: Keep (0, 1, 2) as separate categories and group revonsets > 2 into category 3
        df_dv['revonsets_category'] = df_dv['revonsets']
        df_dv.loc[df_dv['revonsets'] > 2, 'revonsets_category'] = 3
        df_dv['revonsets_category'] = df_dv['revonsets_category'].astype(int)
        df_dv = df_dv.drop(columns=['revonsets']).rename(columns={'revonsets_category': 'revonsets'})

        df_dv = df_dv.sort_values(by=['ccode', 'year']).reset_index(drop=True)

        # Load independent variable dataset
        df_idvs = pd.read_hdf('../../project_data/idvs_augm.h5', key='data').sort_values(by=['ccode', 'year'])

        # Align df_idvs with updated df_dv by ccode (ensures consistency)
        df_idvs = df_idvs[df_idvs['ccode'].isin(df_dv['ccode'])]
        df_idvs = df_idvs.sort_values(by=['ccode', 'year']).reset_index(drop=True)

        # Update both datasets for the new categorization
        df_dv = df_dv.sort_values(by=['ccode', 'year']).reset_index(drop=True)
        df_idvs = df_idvs.sort_values(by=['ccode', 'year']).reset_index(drop=True)

        n_times = len(df_dv['year'].unique())

        common_ccodes = df_dv['ccode'].unique().tolist()
        normalized_df_list = []
        for ccode in common_ccodes:
            idvs_subset = df_idvs[df_idvs['ccode'] == ccode].sort_values(by=['year'])
            float_subset = pd.DataFrame(
                data=self.scaler.fit_transform(idvs_subset[self.float_var].values),
                columns=self.float_var
            )
            normalized_subset = pd.concat(
                [idvs_subset[['ccode', 'year'] + self.category_var].reset_index(drop=True), float_subset],
                axis=1
            )
            normalized_df_list.append(normalized_subset)

        df_idvs = pd.concat(normalized_df_list, axis=0).reset_index(drop=True)

        dv_list, idvs_list = [], []
        for ccode in common_ccodes:
            dv_subset = df_dv[df_dv['ccode'] == ccode].sort_values(by=['year'])
            idvs_subset = df_idvs[df_idvs['ccode'] == ccode].sort_values(by=['year'])

            # Only include if we have complete data for both
            if len(dv_subset) == n_times and len(idvs_subset) == n_times:
                dv_list.append(dv_subset.drop(['ccode', 'year'], axis=1))
                idvs_list.append(idvs_subset.drop(['ccode', 'year'], axis=1))


        assert len(dv_list) == len(idvs_list), f"Mismatch in dimensions: dv_list has {len(dv_list)} elements but idvs_list has {len(idvs_list)} elements"

        # Shape: (n_samples, n_times, n_features)
        XX = np.stack([df.values for df in idvs_list])

        # Shape: (n_samples,) - keeping max value of revonsets only for each ccode
        y = np.array([df['revonsets'].max() for df in dv_list])

        assert XX.shape[0] == len(y), f"Mismatch in dimensions: X has {XX.shape[0]} samples but y has {len(y)} samples"

        return XX, y



    def split_data_fastai(self, y, val_size=0.2, test_size=0.2, random_state=42):
        """
        Split data into stratified train, validation, and test sets.

        Args:
            y (np.array): Target labels for stratification.
            val_size (float): Proportion of data for validation.
            test_size (float): Proportion of data for testing.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: Train, validation, and test indices.
        """
        # Step 1: Split into train + validation and test
        train_indices, test_indices = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Step 2: Split train + validation into train and validation
        train_indices, valid_indices = train_test_split(
            train_indices,
            test_size=val_size,
            stratify=y[train_indices],
            random_state=random_state
        )

        return train_indices, valid_indices, test_indices


    def train_valid_split_data(self):
        """
        Splits the dataset into training, validation, and test sets using FastAI utilities.

        Returns:
            dict: A dictionary containing train, valid, and test splits of X, TDA, and y.
        """
        X, y = self.loading_data()
        train_idx, valid_idx, test_idx = self.split_data_fastai(
            y, val_size=self.val_size, test_size=self.test_size, random_state=42
        )

        # Create data splits
        data_splits = {
            'X_train': X[train_idx], 'y_train': y[train_idx],
            'X_valid': X[valid_idx], 'y_valid': y[valid_idx],
            'X_test': X[test_idx], 'y_test': y[test_idx],
            'splits': (train_idx, valid_idx, test_idx),
        }

        return data_splits

    

    def make_dls(self):
        """
        Creates FastAI-compatible DataLoaders for train/valid and attaches a test DataLoader.

        Returns:
            tuple: (splits, dls)
        """
        sp_data = self.train_valid_split_data()

        train_dl = TSDataLoaderClass(
            sp_data["X_train"], sp_data["y_train"],
            bs=self.bs, shuffle=True
        )
        valid_dl = TSDataLoaderClass(
            sp_data["X_valid"], sp_data["y_valid"],
            bs=self.bs, shuffle=False
        )
        test_dl = TSDataLoaderClass(
            sp_data["X_test"], sp_data["y_test"],
            bs=self.bs, shuffle=False
        )

        dls = DataLoaders(train_dl, valid_dl)

        # Fastai convention: number of input items (usually 1 tensor for time series)
        dls.n_inp = 1

        # Number of classes
        all_y = np.concatenate([sp_data["y_train"], sp_data["y_valid"], sp_data["y_test"]])
        dls.c = int(np.unique(all_y).size)

        # Attach test loader in a simple, explicit way
        dls.test = test_dl

        # Keep references if you need them later
        self._X_test = sp_data["X_test"]
        self._y_test = sp_data["y_test"]

        return sp_data["splits"], dls
    
    def load_arrays_once(self):
        """Load and cache X, y once (reused across CV folds)."""
        if getattr(self, "_cached", None) is None:
            X, y = self.loading_data()
            self._cached = dict(X=X, y=y)
        return self._cached

    @staticmethod
    def make_cv_folds(y, n_splits=5, val_frac=0.15, seed=42):
        """Stratified K-fold over the country/sequence dimension. Each fold ->
        (train_idx, val_idx, test_idx). Same logic/seed as the TDA pipeline, so
        folds are identical when y is identical."""
       
        y = np.asarray(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for tr_full, te in skf.split(np.zeros(len(y)), y):
            tr, va = train_test_split(tr_full, test_size=val_frac,
                                      stratify=y[tr_full], random_state=seed)
            folds.append((np.asarray(tr), np.asarray(va), np.asarray(te)))
        return folds

    def make_dls_fold(self, tr, va, te):
        """Build DataLoaders for one fold from cached arrays + index arrays.
        Mirrors make_dls() exactly (TSDataLoaderClass, n_inp=1, dls.test)."""
        c = self.load_arrays_once()
        X, y = c["X"], c["y"]

        train_dl = TSDataLoaderClass(X[tr], y[tr], bs=self.bs, shuffle=True)
        valid_dl = TSDataLoaderClass(X[va], y[va], bs=self.bs, shuffle=False)
        test_dl  = TSDataLoaderClass(X[te], y[te], bs=self.bs, shuffle=False)

        dls = DataLoaders(train_dl, valid_dl)
        dls.n_inp = 1
        import numpy as np
        dls.c = int(np.unique(np.concatenate([y[tr], y[va], y[te]])).size)
        dls.test = test_dl

        self._X_test = X[te]
        self._y_test = y[te]
        return (tr, va, te), dls

    