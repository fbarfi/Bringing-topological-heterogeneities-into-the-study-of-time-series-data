# file: preparing_data_class.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
        
from fastai.data.core import DataLoaders, DataLoader as FastAIDataLoader
from torch.utils.data import DataLoader as TorchDataLoader


class PDHMultiHelper:
    def __init__(self, dims=(1,2,3,4)):
        self.dims = tuple(dims)
        self.dim_to_id = {int(d): i+1 for i, d in enumerate(sorted(set(self.dims)))}

    @staticmethod
    def _max_points_in_time(remapped_t):
         return sum(a.shape[0] for _, a in remapped_t) if remapped_t else 0

    @staticmethod
    def pad_one_time(remapped_t, max_P, device):
        P = max(1, int(max_P))
        pts = torch.zeros(P, 2, dtype=torch.float32, device=device)
        msk = torch.zeros(P, dtype=torch.float32, device=device)
        did = torch.zeros(P, dtype=torch.long, device=device)  # 0 = PAD
        pos = 0
        for d_id, a in remapped_t:
            a_t = torch.as_tensor(a, dtype=torch.float32, device=device)
            n = min(a_t.shape[0], P - pos)
            if n <= 0: break
            pts[pos:pos+n] = a_t[:n]
            msk[pos:pos+n] = 1.0
            did[pos:pos+n] = int(d_id)  
            pos += n
        return pts, msk, did

    def collate_batch(self, batch):
        T, F = batch[0]["X"].shape
        device = batch[0]["X"].device if isinstance(batch[0]["X"], torch.Tensor) else torch.device("cpu")

        max_P = 1
        remapped_all = []  
        for s in batch:
            remapped_t_list = []
            for t_accum in s["PD_multi"]:  
                remapped_t = []
                for d_raw, arr in t_accum:
                    d_id = self.dim_to_id.get(int(d_raw))
                    if d_id is not None:
                        remapped_t.append((d_id, arr))
                remapped_t_list.append(remapped_t)
                max_P = max(max_P, self._max_points_in_time(remapped_t))
            remapped_all.append(remapped_t_list)

        Xs = torch.stack([torch.as_tensor(s["X"]) for s in batch], 0).to(device)  # [B,T,F]

        PD_pts_list, PD_mask_list, PD_dim_list = [], [], []
        for remapped_t_list in remapped_all:
            pts_t, msk_t, did_t = [], [], []
            for remapped_t in remapped_t_list:
                pts, msk, did = self.pad_one_time(remapped_t, max_P=max_P, device=Xs.device)
                pts_t.append(pts); msk_t.append(msk); did_t.append(did)
            PD_pts_list.append(torch.stack(pts_t, 0))
            PD_mask_list.append(torch.stack(msk_t, 0))
            PD_dim_list.append(torch.stack(did_t, 0))

        PD_pts  = torch.stack(PD_pts_list, 0)   # [B,T,P,2]

        PD_mask = torch.stack(PD_mask_list, 0)  # [B,T,P]
        PD_dim  = torch.stack(PD_dim_list, 0)   # [B,T,P]
        y = torch.as_tensor([s["y"] for s in batch], dtype=torch.long, device=Xs.device)
        return Xs, PD_pts, PD_mask, PD_dim, y


class TSDatasetPDHMulti(Dataset):
    """
    torch Dataset returning dicts consumed by the collate:
    {"X": [T,F], "PD_multi": list[T] of list[(raw_dim:int, ndarray[n,2])], "y": int}
    """
    def __init__(self, X, PD_list, y, dims=(1,2,3,4)):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.PD = PD_list
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.dims = tuple(dims)
        assert len(self.X) == len(self.PD) == len(self.y)
        T0 = self.X.shape[1]
        for i, tlist in enumerate(self.PD):
            if len(tlist) != T0:
                raise ValueError(f"Sample {i} PD length {len(tlist)} != X.T {T0}")

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx):
        Xi = self.X[idx]
        Yi = int(self.y[idx].item())
        PDti = self.PD[idx]  
        clean = []
        for t_dict in PDti:
            t_accum = []
            for d, arr in t_dict.items():
                if arr is None:
                    continue
                a = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
                if a.size == 0:
                    continue
                t_accum.append((int(d), a))
            clean.append(t_accum)
        return {"X": Xi, "PD_multi": clean, "y": Yi}

    def new_empty(self):
        """Return a loader with an empty dataset but identical schema/params."""
        dims = self.dims
        X_empty = self.X[:0]               # shape (0, T, F)
        y_empty = self.y[:0]               # shape (0,)
        PD_empty = self.PD[:0]             # list length 0
        ds_empty = TSDatasetPDHMulti(X_empty, PD_empty, y_empty, dims=dims)
        return TSDataLoaderPDHMulti(ds_empty)


class TSDataLoaderPDHMulti(TorchDataLoader):
    def __init__(self, 
                 dataset: TSDatasetPDHMulti, 
                 batch_size=8, 
                 shuffle=True, 
                 drop_last=False, 
                 device=None, 
                 **kwargs):
        if 'bs' in kwargs:
            batch_size = int(kwargs.pop('bs'))
        kwargs.pop('with_labels', None)

        self.helper = PDHMultiHelper(dims=getattr(dataset, "dims", (1,2,3,4)))
        self.device = device
        self.bs = int(batch_size)
        n_items = len(dataset)
        self.shuffle = bool(shuffle) if n_items > 0 else False 
        self.drop_last = bool(drop_last)

        super().__init__(
            dataset,
            batch_size=self.bs,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            collate_fn=self.helper.collate_batch,
            **kwargs,
        )
        self.n = len(dataset)
        self.c = int(torch.unique(dataset.y).numel()) if self.n else 0

    def to(self, device):
        self.device = device
        return self

    def one_batch(self):
        return next(iter(self))

    def new(self, dataset=None, **kwargs):
        """fastai DataLoader.new compatibility; sanitize kwargs."""
        dataset = dataset or self.dataset
        params = dict(batch_size=self.bs, 
                      shuffle=self.shuffle, 
                      drop_last=self.drop_last, 
                      device=self.device)
        params.update(kwargs)

        if 'bs' in params:
            params['batch_size'] = int(params.pop('bs'))
        params.pop('with_labels', None)

        return TSDataLoaderPDHMulti(dataset, **params)


class PreparingDataClass:
    def __init__(self, args):
        self.n_years = args['n_years']
        self.float_var = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity']
        self.category_var = ['revleader', 'new_oil_state', 'coldwar', 'majorpower']
        self.val_size = args['val_size']
        self.test_size = args['test_size']
        self.scaler = StandardScaler()
        self.pd_folder = args['pd_folder']
        self.Hidx = args['Hidx']
        self.dims = tuple(args.get('dims', (1,2,3,4)))
        self.bs = int(args.get('bs', 8))
        self.device = args.get('device', None)
        self.shuffle_train = bool(args.get('shuffle_train', True))

    def _load_pd_file(self):
        """
        Returns: PD_all: list length T where each item is a dict {dim -> ndarray[n,2]}.
        Assumes CSV per year: PD_H_{Hidx}_{year_idx}.csv with cols ['b','d','h_dim'].
        """
        PD_all = []
      
        for year_idx,_ in enumerate(self.n_years):
            path = os.path.join(self.pd_folder, f"PD_H_{self.Hidx}_{year_idx}.csv")
            df = pd.read_csv(path)
            dct = {}
            if {"b", "d", "h_dim"}.issubset(df.columns):
                g = df.groupby("h_dim")
                for d in self.dims:
                    if d in g.groups:
                        bd = g.get_group(d)[["b", "d"]].to_numpy(dtype=np.float32, copy=True)
                        dct[int(d)] = bd
            PD_all.append(dct)
        return PD_all

    def _load_align(self):
        dv = pd.read_hdf('../../project_data/dv_augm.h5', key='data').sort_values(['ccode','year'])
        dv['revonsets_category'] = dv['revonsets']
        dv.loc[dv['revonsets'] > 2, 'revonsets_category'] = 3
        dv['revonsets'] = dv['revonsets_category'].astype(int)
        dv = dv.drop(columns=['revonsets_category']).reset_index(drop=True)

        idv = pd.read_hdf('../../project_data/idvs_augm.h5', key='data').sort_values(['ccode','year'])
                # .drop(['peaceyears','iraniraq88'], axis=1) \
                # .sort_values(['ccode','year'])
        idv = idv[idv['ccode'].isin(dv['ccode'])].reset_index(drop=True)

        n_times = len(dv['year'].unique())
        ccodes = dv['ccode'].unique().tolist()

        norm_rows = []
        for c in ccodes:
            sub = idv[idv['ccode']==c].sort_values('year')
            floats = pd.DataFrame(StandardScaler().fit_transform(sub[self.float_var].values),
                                  columns=self.float_var)
            norm_rows.append(pd.concat([sub[['ccode','year']+self.category_var].reset_index(drop=True),
                                        floats], axis=1))
        idv = pd.concat(norm_rows, axis=0).reset_index(drop=True)

        dv_list, idv_list, years_list = [], [], []
        for c in ccodes:
            dvc = dv[dv['ccode']==c].sort_values('year')
            ivc = idv[idv['ccode']==c].sort_values('year')
            if len(dvc)==n_times and len(ivc)==n_times:
                dv_list.append(dvc.drop(['ccode','year'], axis=1))
                idv_list.append(ivc.drop(['ccode','year'], axis=1))
                years_list.append(dvc['year'].values.astype(int))

        X = np.stack([df.values for df in idv_list])                 # [B,T,F]
        y = np.array([df['revonsets'].max() for df in dv_list])      # [B]
        years = np.stack(years_list)                                  # [B,T]

        timeline = np.unique(dv['year'].values)                       
        years_idx = np.searchsorted(timeline, years).astype(int)
        return X, y, years_idx

    @staticmethod
    def _split(y, val_size=0.2, test_size=0.1, seed=42):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        tr, va = train_test_split(tr, test_size=val_size, stratify=y[tr], random_state=seed)
        return tr, va, te

    def prepare(self):
        X, y, years_idx = self._load_align()     
        PD_all = self._load_pd_file()             
        B, T = X.shape[0], X.shape[1]
        assert len(PD_all) == T, f"PD_all length {len(PD_all)} != X.T {T}"

        tr, va, te = self._split(y, self.val_size, self.test_size, 42)
        print(f'--------------tr:{len(tr)}, va : {len(va)}, te : {len(te)}')
    

        PD_train = [PD_all[:] for _ in range(X[tr].shape[0])]
        PD_valid = [PD_all[:] for _ in range(X[va].shape[0])]
        PD_test  = [PD_all[:] for _ in range(X[te].shape[0])]
        
        return {
            "X_train": X[tr], "y_train": y[tr], "PD_train": PD_train,
            "X_valid": X[va], "y_valid": y[va], "PD_valid": PD_valid,
            "X_test":  X[te], "y_test":  y[te], "PD_test":  PD_test,
            "splits": (tr, va, te),
            "dims": self.dims,
        }

    def make_dls(self):
        data = self.prepare()
        dims = data["dims"]

        ds_tr = TSDatasetPDHMulti(data["X_train"], data["PD_train"], data["y_train"], dims=dims)
        ds_va = TSDatasetPDHMulti(data["X_valid"], data["PD_valid"], data["y_valid"], dims=dims)
        ds_te = TSDatasetPDHMulti(data["X_test"],  data["PD_test"],  data["y_test"],  dims=dims)

        dl_train = TSDataLoaderPDHMulti(ds_tr, batch_size=self.bs, shuffle=True,  drop_last=False, device=self.device)
        dl_valid = TSDataLoaderPDHMulti(ds_va, batch_size=self.bs, shuffle=False, drop_last=False, device=self.device)

        dls = DataLoaders(dl_train, dl_valid)
        dls.n_inp = 4
        dls.c = int(torch.unique(ds_tr.y).numel())

        dl_test = dls.test_dl(ds_te, with_labels=True, bs=self.bs, shuffle=False, device=self.device)

        dls.test = dl_test
        self._ds_test = ds_te
        self._y_test = data["y_test"]

        return data["splits"], dls
    
    def load_arrays_once(self):
        """Load and cache X, y, PD_all a single time (for CV reuse)."""
        if getattr(self, "_cached", None) is None:
            X, y, years_idx = self._load_align()
            PD_all = self._load_pd_file()
            assert len(PD_all) == X.shape[1], \
                f"PD_all length {len(PD_all)} != X.T {X.shape[1]}"
            self._cached = dict(X=X, y=y, PD_all=PD_all)
        return self._cached

    def make_dls_fold(self, tr, va, te):
        """
        Build DataLoaders for one fold from cached arrays and explicit index
        arrays (over the country/sequence dimension). Mirrors make_dls() exactly
        in dataset/loader construction and the dls.test attachment, so the rest
        of ModelsClass is unchanged.
        """
        c = self.load_arrays_once()
        X, y, PD_all = c["X"], c["y"], c["PD_all"]

        def _pd_for(n):  # same per-sample PD replication make_dls() uses
            return [PD_all[:] for _ in range(n)]

        ds_tr = TSDatasetPDHMulti(X[tr], _pd_for(len(tr)), y[tr], dims=self.dims)
        ds_va = TSDatasetPDHMulti(X[va], _pd_for(len(va)), y[va], dims=self.dims)
        ds_te = TSDatasetPDHMulti(X[te], _pd_for(len(te)), y[te], dims=self.dims)

        dl_train = TSDataLoaderPDHMulti(ds_tr, batch_size=self.bs, shuffle=True,
                                        drop_last=False, device=self.device)
        dl_valid = TSDataLoaderPDHMulti(ds_va, batch_size=self.bs, shuffle=False,
                                        drop_last=False, device=self.device)

        dls = DataLoaders(dl_train, dl_valid)
        dls.n_inp = 4
        dls.c = int(torch.unique(ds_tr.y).numel())

        dl_test = dls.test_dl(ds_te, with_labels=True, bs=self.bs,
                              shuffle=False, device=self.device)
        dls.test = dl_test
        self._ds_test = ds_te
        self._y_test = y[te]
        return (tr, va, te), dls

    @staticmethod
    def make_cv_folds(y, n_splits=5, val_frac=0.15, seed=42):
        """
        Stratified K-fold over the sequence (country) dimension. Each fold
        yields (train_idx, val_idx, test_idx): one fold is the test set, a
        stratified slice of the remainder is validation, the rest is train.
        """
        
        y = np.asarray(y)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for tr_full, te in skf.split(np.zeros(len(y)), y):
            # carve a stratified validation slice out of the training portion
            tr, va = train_test_split(tr_full, test_size=val_frac,
                                      stratify=y[tr_full], random_state=seed)
            folds.append((np.asarray(tr), np.asarray(va), np.asarray(te)))
        return folds
