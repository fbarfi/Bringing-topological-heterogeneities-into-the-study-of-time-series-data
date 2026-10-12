from __future__ import annotations
from typing import Dict, Iterable, List, Any, Set, Tuple

from collections import Counter
import os
import csv
import pandas as pd 
import numpy as np
import gudhi as gd
import json
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.sklearn.rips_persistence import RipsPersistence
from gudhi.representations import DiagramSelector, DiagramScaler

from sklearn.preprocessing  import MinMaxScaler


class GudhiPersistenceDiagramsClass():
    def __init__(self,args):

        self.args = args
        self.dataset_file = '../../project_data/dataset_augm.csv'
        self.homology_dimensions = args['homology_dimensions']
        self.Hidx = int(''.join(map(str, self.homology_dimensions[1:])))
        self.max_dimension = args['max_dimension']
        self.float_vars = ['logoil', 'ln_gdppc', 'lnpop', 'muslim', 'borders', 'polity']
        self.binary_vars = ['revleader', 'new_oil_state', 'coldwar', 'majorpower']
        self.n_years = args['n_years']
        self.folder = args['folder']
        self.PersistenceDiagrams = args['PersistenceDiagrams']
        self.PersistenceLandscapes = args['PersistenceLandscapes']
        self.args['embedding_metadata_file'] = os.path.join(args['folder'], 'embedding_metadata.json')


    def data_preprocessing(self):
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        df = pd.read_csv(self.dataset_file).sort_values(by = ['year','ccode']).reset_index()   
        df_y = df[['year','ccode','revonsets']]
        df = df.drop(columns=['revonsets'])
        df_X = df[['year','ccode']+ self.binary_vars + self.float_vars]

        print(f'all_covariates: {df.columns.values[2:].tolist()}') 
        self.args['all_covariates'] = df.columns.values[2:].tolist()

        df_X_float = pd.DataFrame(data = self.scaler.fit_transform(df_X[self.float_vars].values),
                                 columns = self.float_vars).reset_index(drop=True)
        df_X = pd.concat([df_X[['year','ccode'] +self.binary_vars], df_X_float], axis=1)

        X_list = [df_X.loc[df_X['year'] == year,:].iloc[:,2:].reset_index(drop=True).values
                  for year in df_X['year'].unique().tolist()]
        y_list = [df_y.loc[df_y['year'] == year,:].iloc[:,-1].reset_index(drop=True)
                  for year in df_y['year'].unique().tolist()]
        
        return df_X, X_list,y_list


    def _embed_1d(self, x: np.ndarray, D: int, tau: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size == 0:
            raise ValueError("empty series for TDE")
        tde = TimeDelayEmbedding(dim=int(D), delay=int(tau), skip=1)
        X_emb = tde.transform(x)
        X_emb = np.asarray(X_emb, dtype=float)
        expected_m = x.shape[0] - (int(D) - 1) * int(tau)
        if X_emb.ndim != 2 or X_emb.shape != (expected_m, int(D)):
            raise ValueError(f"TDE shape mismatch: got {X_emb.shape}, expected {(expected_m, int(D))}")
        return X_emb

    def _embed_columns_list(self, X_pad: np.ndarray, D: int, tau: int) -> np.ndarray:
        """Gudhi TDE on a LIST of column series; returns (m, n_vars*D)."""
        X_pad = np.asarray(X_pad, dtype=float)
        if X_pad.ndim != 2:
            raise ValueError(f"X_pad must be 2D, got {X_pad.ndim}D")
        n_rows, n_vars = X_pad.shape

        series_list = [X_pad[:, j].reshape(-1) for j in range(n_vars)]

        tde = TimeDelayEmbedding(dim=int(D), delay=int(tau), skip=1)
        emb_list = tde.transform(series_list)  

        emb_list = [np.asarray(e, dtype=float) for e in emb_list]
        if len(emb_list) != n_vars:
            raise ValueError(f"TDE returned {len(emb_list)} embeddings, expected {n_vars}")
        m = emb_list[0].shape[0]
        for j, e in enumerate(emb_list):
            if e.ndim != 2 or e.shape[1] != int(D) or e.shape[0] != m:
                raise ValueError(f"Column {j}: got {e.shape}, expected (m={m}, D={D})")

        return np.hstack(emb_list)


    def _validate_vertices_ccodes(
            self,
            df_X: pd.DataFrame,
            years_block: List[Dict[str, Any]],
            strict: bool = True,
            sample: int = 10,
        ) -> None:
        """
        Validate per-year one-to-one coverage:
        - every ccode in df_X[year] appears exactly once in vertices
        - no duplicates, no missing.
        """
        problems: List[str] = []
        base = df_X[["year", "ccode"]].copy()
        base["year"] = base["year"].astype(int)
        base["ccode"] = base["ccode"].astype(int)

        for entry in years_block:
            y = int(entry["year"])
            verts = entry.get("vertices", [])
            got_cc = [int(v.get("ccode", -1)) for v in verts]

            dfy = base[base["year"] == y]
            exp_cc = dfy["ccode"].tolist()

            got_counter = Counter(got_cc)
            exp_counter = Counter(exp_cc)

            dup_cc = [cc for cc, c in got_counter.items() if c > 1]
            if dup_cc:
                problems.append(f"[{y}] duplicate ccodes in vertices: {dup_cc[:sample]} (total {len(dup_cc)})")

            missing = list((exp_counter - got_counter).elements())
            extra   = list((got_counter - exp_counter).elements())
            if missing or extra:
                if missing:
                    problems.append(f"[{y}] missing ccodes: {sorted(set(missing))[:sample]} (total {len(set(missing))})")
                if extra:
                    problems.append(f"[{y}] extra ccodes: {sorted(set(extra))[:sample]} (total {len(set(extra))})")

            if len(got_cc) != len(exp_cc):
                problems.append(f"[{y}] size mismatch: vertices={len(got_cc)} vs df_X={len(exp_cc)}")

        if problems:
            msg = "Vertex–ccode validation failed:\n" + "\n".join(problems)
            if strict:
                raise ValueError(msg)
            else:
                print("[warn]", msg)


    def calculating_multiple_embedding(self) -> List[np.ndarray]:
        """Pad-first TDE so no ccode rows are dropped per year; vertices carry t_index & ccode."""
        df_X, X_list, _ = self.data_preprocessing()
        print(f"original data: X_list[0].shape: {X_list[0].shape}")

        n_years: List[int] = list(self.n_years)
        if len(n_years) != len(X_list):
            raise ValueError("Year/X length mismatch")
        if "year" not in df_X.columns or "ccode" not in df_X.columns:
            raise ValueError("df_X must contain 'year' and 'ccode'")

        variables: List[str] = df_X.columns[2:].tolist()
        if not variables:
            raise ValueError("No variables after ['year','ccode']")
        n_vars = len(variables)

        delay: int = int(self.args.get("delay", 1))
        emb_dim: int = int(self.args.get("emb_dim", 30))
        pad_rows: int = (emb_dim - 1) * delay

        out_path = self.args.get("embedding_metadata_file")
        if not out_path:
            raise ValueError("embedding_metadata_file not set")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        def _ccodes_for_year(y: int) -> List[int]:
            s = pd.Series(df_X[df_X["year"] == y]["ccode"].tolist(), dtype="Int64")
            return s.astype(int).tolist()

        feature_catalog: List[Tuple[int, str, int]] = []
        for v_idx, var_name in enumerate(variables):
            for lag in range(emb_dim):
                feat_id = v_idx * emb_dim + lag
                feature_catalog.append((feat_id, var_name, lag))

        X_te_list: List[np.ndarray] = []
        years_block: List[Dict[str, Any]] = []

        for i, y in enumerate(n_years):
            X = np.asarray(X_list[i])  # (n_states, n_vars)
            if X.ndim != 2 or X.shape[1] != n_vars:
                raise ValueError(f"Year {y}: X shape {X.shape} incompatible with n_vars={n_vars}")
            n_states = int(X.shape[0])

            # Pad-front to preserve all n_states after TDE
            if pad_rows > 0:
                take = min(pad_rows, n_states)
                pad_block = X[:take, :]
                if take < pad_rows:
                    reps = (pad_rows + take - 1) // take
                    pad_block = np.vstack([pad_block] * reps)[:pad_rows, :]
                X_pad = np.vstack([pad_block, X])
            else:
                X_pad = X

            X_te = self._embed_columns_list(X_pad, emb_dim, delay)  # (m, n_vars*emb_dim)

            if X_te.shape[0] != n_states:
                raise ValueError(f"Year {y}: expected rows={n_states}, got {X_te.shape[0]}")
            if X_te.shape[1] != n_vars * emb_dim:
                raise ValueError(f"Year {y}: expected cols={n_vars*emb_dim}, got {X_te.shape[1]}")
            X_te_list.append(X_te)

            ccodes = _ccodes_for_year(int(y))
            if len(ccodes) != n_states:
                raise ValueError(f"Year {y}: ccode rows ({len(ccodes)}) != n_states ({n_states})")

            vertices = [{"id": v, "t_index": v, "ccode": int(ccodes[v])} for v in range(n_states)]
            features = [{"id": f_id, "variable": var_name, "lag": lag} for (f_id, var_name, lag) in feature_catalog]
            years_block.append({"year": int(y), "vertices": vertices, "features": features})

        self._validate_vertices_ccodes(df_X, years_block, strict=True)

        payload = {
            "schema": "embedding_metadata_v1",
            "emb_dim": int(emb_dim),
            "n_vars": int(n_vars),
            "variables": variables,
            "homology_dimensions": list(map(int, self.homology_dimensions)),  
            "n_years": years_block,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"embedded data: X_te_list[0].shape: {X_te_list[0].shape}")

        years = [entry["year"] for entry in years_block]
        save_path = os.path.join(self.folder, "embedded_points.npz")
        np.savez_compressed(
            save_path,
            **{f"y_{int(y)}": np.asarray(X_te_list[i], dtype=float) for i, y in enumerate(years)}
        )

        return X_te_list


    """
    Computing diagrams
    """
    def rips(self,X):
        rips_pers = RipsPersistence(homology_dimensions = self.homology_dimensions,
                                     input_type = 'point cloud') 
        diagrams = rips_pers.fit_transform([X])

        selector = DiagramSelector(use = True, limit=np.inf, point_type="finite")
        diagram = selector.fit_transform(diagrams[0])
        scaler = DiagramScaler(use = True, scalers=[([0,1], MinMaxScaler())])
        diagram = scaler.fit_transform(diagram)
        return diagram

    def computing_one_PD(self, X, idx):
        PD = self.rips(X)  
        frames = []
        for i, diag in enumerate(PD):
            dim = int(self.homology_dimensions[i])  
            df = pd.DataFrame(diag, columns=["b", "d"])
            df["h_dim"] = dim
            frames.append(df)

        df_H = pd.concat(frames, ignore_index=True)
        filepd = os.path.join(self.PersistenceDiagrams, f"PD_H_{self.Hidx}_{idx}.csv")
        df_H.to_csv(filepd, index=False)
        return df_H


    def grouping_PDs(self):
        
        X_list = self.calculating_multiple_embedding()

        for idx, X in enumerate(X_list):
            self.computing_one_PD(X, idx)
        
        print(f"Done saving all PDs")

        return 

   