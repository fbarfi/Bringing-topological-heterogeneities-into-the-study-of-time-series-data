from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tsai.all import load_learner

from .preparing_data_class import PreparingDataClass


class PDDeepSetsPlots:
    """
    Standalone PDDeepSets visualizer 

    Time axis default: 1945..2010 inclusive (length 66).
    If your sequence length differs, pass `years=` to run().
    """

    def __init__(self, args: Dict):
        self.args = args
        self.folder = str(args["folder"])
        self.model_path = str(args["model_path"])
        self.bs = int(args.get("bs", 8))
        self.device = args.get("device", torch.device("cpu"))

        os.makedirs(self.folder, exist_ok=True)

        # Prefer the fold's prebuilt loaders (prep, splits, dls) if the caller
        # passed them in args["_prebuilt"], so the viz operates on the SAME
        # held-out split as training/eval rather than rebuilding a fresh one.
        prebuilt = args.get("_prebuilt", None)
        if prebuilt is not None:
            self.data_class = prebuilt[0]
            self.splits = prebuilt[1]
            self.dls = prebuilt[2]
        else:
            self.data_class = PreparingDataClass(args)
            self.splits, self.dls = self.data_class.make_dls()

        self.dls_train = self.dls[0]
        self.dls_valid = self.dls[1]
        self.dls_test = getattr(self.dls, "test", None)
        if self.dls_test is None:
            ds_te = self.data_class._ds_test
            self.dls_test = self.dls.test_dl(
                ds_te, with_labels=True, bs=self.bs, shuffle=False, device=self.device
            )

        self.learn = None
        self.model = None

    # ------------------------- model/dataloader -------------------------
    def load_model(self):
        path = Path(self.model_path + "trained_classification_model.pkl")
        self.learn = load_learner(path, cpu=True)
        self.model = self.learn.model.to(self.device).eval()
        return self.learn

    def _get_dl(self, which: str):
        which = which.lower().strip()
        if which == "train":
            return self.dls_train
        if which in ("valid", "val"):
            return self.dls_valid
        if which == "test":
            return self.dls_test
        raise ValueError("which must be one of: 'train', 'valid', 'test'")

    @staticmethod
    def _nth_batch(dl, batch_idx: int):
        for i, b in enumerate(dl):
            if i == batch_idx:
                return b
        raise IndexError(f"batch_idx={batch_idx} out of range for this dataloader.")

    # ------------------------- PDDeepSets internals -------------------------
    @torch.no_grad()
    def pd_enc_intermediates(self, pd_enc, PD_pts, PD_mask, PD_dim) -> Dict[str, torch.Tensor]:
        # Use the encoder's own attention_weights() so the diagnostics read the
        # exact weights the model uses (persistence prior + padding-aware
        # standardization + temperature + masking). Re-deriving the softmax here
        # would drift from training whenever the encoder forward changes.
        base, _pers = pd_enc._prep_base(PD_pts, PD_mask)
        dim_e = pd_enc.emb(PD_dim.clamp_min(0))
        x_in = torch.cat([base, dim_e], dim=-1)
        feats = pd_enc.phi(x_in)

        w2d = pd_enc.attention_weights(PD_pts, PD_mask, PD_dim)   # [B,T,P]

        tau = pd_enc.tau.clamp_min(1e-3) if isinstance(pd_enc.tau, torch.Tensor) else torch.tensor(
            max(1e-3, float(pd_enc.tau))
        )

        w = w2d.unsqueeze(-1)                                     # [B,T,P,1] (contract preserved)
        agg = (feats * w).sum(dim=2)
        out = pd_enc.post(agg)

        return {"base": base,
                "feats": feats,
                "logits": None,
                "tau": tau.detach().clone(),
                "w": w,
                "agg": agg,
                "out": out}

    # ------------------------- utilities -------------------------
    @staticmethod
    def _savefig(path: str):
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=160)
        plt.close()

    @staticmethod
    def _valid_at_time(PD_pts_bt, PD_mask_bt, PD_dim_bt, w_bt):
        m = (PD_mask_bt > 0)
        if m.sum().item() == 0:
            return None
        pts = PD_pts_bt[m]
        dims = PD_dim_bt[m]
        w = w_bt[m]
        pers = (pts[:, 1] - pts[:, 0]).clamp_min(0)
        return pts, dims, w, pers

    @staticmethod
    def _attention_entropy(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        w = w.clamp_min(eps)
        return -(w * w.log()).sum(dim=-1)

    @staticmethod
    def _effective_num_points(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return 1.0 / (w.pow(2).sum(dim=-1).clamp_min(eps))

    @staticmethod
    def _default_years(T: int) -> np.ndarray:
        years = np.arange(1950, 2011, dtype=int)  # 1950..2010 inclusive
        if len(years) != T:
            # fallback: still label meaningfully if T differs
            return np.arange(T, dtype=int)
        return years

    # ------------------------- plots -------------------------
    def plot_attention_mass_by_dim_over_time(self, bb, PD_mask, PD_dim, w, years: np.ndarray, 
                                             batch=0, out_path: str = ""):
        mask = PD_mask[batch].detach().cpu()              # [T,P]
        dims = PD_dim[batch].detach().cpu()               # [T,P]
        ww = w[batch, :, :, 0].detach().cpu()             # [T,P]

        valid_dims = sorted([int(d) for d in torch.unique(dims[mask > 0]).tolist() if int(d) != 0])
        if not valid_dims:
            return

        T = ww.shape[0]
        mass = {d: np.zeros(T, dtype=np.float32) for d in valid_dims}
        for t in range(T):
            m_t = mask[t] > 0
            if m_t.sum().item() == 0:
                continue
            w_t = ww[t][m_t]
            d_t = dims[t][m_t]
            w_t = w_t / (w_t.sum() + 1e-12)
            for d in valid_dims:
                mass[d][t] = float(w_t[d_t == d].sum().item())

        plt.figure(figsize=(10, 4))
        years_x = np.asarray(years)[:T] if len(np.asarray(years)) >= T else np.arange(T)
        for d in valid_dims:
            plt.plot(years_x, mass[d][:len(years_x)], marker="o", linewidth=1.5, label=f"dim={d}")
        plt.xlabel("year")
        plt.ylabel("conditioning attention mass")
        plt.title(f"{bb}:Conditioning attention mass by homology dim over time (batch={batch})")
        plt.ylim(0.0, 1.0)
        plt.legend(ncol=min(4, len(valid_dims)))
        if out_path:
            self._savefig(out_path)


    def plot_attention_sharpness_over_time(self,bb, PD_mask, w, years: np.ndarray, batch=0, out_path: str = ""):
        mask = PD_mask[batch].detach().cpu()              # [T,P]
        ww = w[batch, :, :, 0].detach().cpu()             # [T,P]
        T = ww.shape[0]
        entropy = np.zeros(T, dtype=np.float32)
        effn = np.zeros(T, dtype=np.float32)
        for t in range(T):
            m_t = mask[t] > 0
            if m_t.sum().item() == 0:
                continue
            w_t = ww[t][m_t]
            w_t = w_t / (w_t.sum() + 1e-12)
            entropy[t] = float(self._attention_entropy(w_t).item())
            effn[t] = float(self._effective_num_points(w_t).item())

        fig, ax_left = plt.subplots(figsize=(10, 4))
        years_x = np.asarray(years)[:T] if len(np.asarray(years)) >= T else np.arange(T)
        entropy = entropy[:len(years_x)]
        effn = effn[:len(years_x)]

        # Left axis: entropy
        color_ent = "tab:blue"
        l1, = ax_left.plot(years_x, entropy, marker="o", linewidth=1.5,
                        color=color_ent, label="entropy (lower=sharper)")
        ax_left.set_xlabel("year")
        ax_left.set_ylabel("entropy (nats)", color=color_ent)
        ax_left.tick_params(axis="y", labelcolor=color_ent)

        # Right axis: effective #points
        ax_right = ax_left.twinx()
        color_eff = "tab:orange"
        l2, = ax_right.plot(years_x, effn, marker="o", linewidth=1.5,
                            color=color_eff, label="effective #points")
        ax_right.set_ylabel("effective #points", color=color_eff)
        ax_right.tick_params(axis="y", labelcolor=color_eff)

        ax_left.set_title(f"{bb}:Conditioning attention sharpness over time (batch={batch})")
        ax_left.legend(handles=[l1, l2], loc="upper left")

        fig.tight_layout()
        if out_path:
            self._savefig(out_path)


    @torch.no_grad()
    def _tda_prior_intermediates(
        self,
        which: str = "test",
        batch_idx: int = 0,
        sample_in_batch: int = 0,
        years: np.ndarray | None = None,
    ):
        """
        Collects interpretable intermediates for TDATimePrior + PoE on ONE sample:
        - H_tda: [T, tda_dim]
        - mu_t, lv_t: [T, latent_dim]
        - sigma_t: [T, latent_dim]
        - poe_mu, poe_lv: [latent_dim]
        - normalized precision weights over time: [T, latent_dim]
        """
        if self.model is None:
            self.load_model()

        dl = self._get_dl(which)
        x, pd_pts, pd_mask, pd_dim, y = self._nth_batch(dl, batch_idx)

        pd_pts = pd_pts.to(self.device, dtype=torch.float32)
        pd_mask = pd_mask.to(self.device, dtype=torch.float32)
        pd_dim = pd_dim.to(self.device, dtype=torch.long)

        # --- 1) H_tda from PD encoder ---
        H_tda = self.model.pd_enc(pd_pts, pd_mask, pd_dim)  # [B,T,tda_dim]
        H = H_tda[sample_in_batch]                          # [T,tda_dim]

        # --- 2) Per-time prior params ---
        mu_t, lv_t = self.model.prior(H.unsqueeze(0))        # [1,T,D]
        mu_t = mu_t.squeeze(0)                               # [T,D]
        lv_t = lv_t.squeeze(0)                               # [T,D]
        sigma_t = (0.5 * lv_t).exp()                         # [T,D]

        # --- 3) PoE pooling (your staticmethod) ---
        poe_mu, poe_lv = self.model.prior.poe(mu_t.unsqueeze(0), lv_t.unsqueeze(0))  # [1,D]
        poe_mu = poe_mu.squeeze(0)                            # [D]
        poe_lv = poe_lv.squeeze(0)                            # [D]
        poe_sigma = (0.5 * poe_lv).exp()

        # --- 4) Precision weights over time (normalized, per latent dim) ---
        var_t = lv_t.exp() + 1e-6                             # [T,D]
        prec_t = 1.0 / var_t                                  # [T,D]
        prec_norm = prec_t / (prec_t.sum(dim=0, keepdim=True) + 1e-6)  # [T,D]

        T = mu_t.shape[0]
        if years is None:
            years = np.arange(1950, 2011, dtype=int) if T == 61 else np.arange(T, dtype=int)
        years = np.asarray(years, dtype=int)
        if len(years) != T:
            raise ValueError(f"years length ({len(years)}) must match T ({T}).")

        return {
            "years": years,
            "H_tda": H.detach().cpu(),
            "mu_t": mu_t.detach().cpu(),
            "lv_t": lv_t.detach().cpu(),
            "sigma_t": sigma_t.detach().cpu(),
            "poe_mu": poe_mu.detach().cpu(),
            "poe_lv": poe_lv.detach().cpu(),
            "poe_sigma": poe_sigma.detach().cpu(),
            "prec_norm": prec_norm.detach().cpu(),
            "y": int(y[sample_in_batch].detach().cpu().item()),
        }


    def save_tda_prior(
        self,
        bb: str = "bb",
        which: str = "test",
        batch_idx: int = 0,
        sample_in_batch: int = 0,
        years_to_show: tuple[int, int, int, int] = (1955, 1975, 1995, 2010),
        top_latent_dims: int = 6,
        prefix: str = "tdaprior",
        years: np.ndarray | None = None,
    ) -> Dict[str, str]:
        """
        Creates interpretable plots for TDATimePrior + PoE and saves into args["folder"].

        Outputs:
        1) A single figure with:
            - top latent dims: mu_t over years (lines) with a dashed poe_mu reference
            - precision mass over years for those dims (lines)
        2) A 4-year “snapshot” bar figure showing, for each selected year:
            - mu_t[year] vs poe_mu (for top dims)
            - sigma_t[year] vs poe_sigma (for top dims)
        3) CSV with per-year precision mass aggregated over top dims

        Returns dict of saved paths.
        """
        pack = self._tda_prior_intermediates(
            which=which, batch_idx=batch_idx, sample_in_batch=sample_in_batch, years=years
        )
        yrs = pack["years"]
        mu_t = pack["mu_t"].numpy()           # [T,D]
        sig_t = pack["sigma_t"].numpy()       # [T,D]
        poe_mu = pack["poe_mu"].numpy()       # [D]
        poe_sig = pack["poe_sigma"].numpy()   # [D]
        prec = pack["prec_norm"].numpy()      # [T,D]

        T, D = mu_t.shape

        var_by_dim = mu_t.var(axis=0)
        sel = np.argsort(var_by_dim)[::-1][: min(top_latent_dims, D)]

        out_main = os.path.join(self.folder, f"{prefix}_{which}_b{batch_idx}_s{sample_in_batch}_mu_and_precision.png")
        out_snap = os.path.join(self.folder, f"{prefix}_{which}_b{batch_idx}_s{sample_in_batch}_year_snapshots.png")
        out_csv  = os.path.join(self.folder, f"{prefix}_{which}_b{batch_idx}_s{sample_in_batch}_precision_mass_topdims.csv")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # mu_t lines
        for j in sel:
            axs[0].plot(yrs, mu_t[:, j], marker="o", linewidth=1.5, label=f"z{j}: mu_t")
            axs[0].axhline(poe_mu[j], linestyle="--", linewidth=1.2)

        axs[0].set_ylabel("prior mean (mu_t)")
        axs[0].set_title(f"{bb}: TDATimePrior: per-year prior means (top latent dims) + PoE pooled mean (dashed)")
        axs[0].legend(ncol=2, fontsize=8)

        # precision mass lines
        for j in sel:
            axs[1].plot(yrs, prec[:, j], marker="o", linewidth=1.5, label=f"z{j}: precision share")

        axs[1].set_xlabel("year")
        axs[1].set_ylabel("precision share (per dim)")
        axs[1].set_title(f"{bb}: PoE: which years dominate (normalized precision weights per latent dim)")
        axs[1].legend(ncol=2, fontsize=8)

        plt.tight_layout()
        plt.savefig(out_main, bbox_inches="tight", dpi=180)
        plt.close(fig)

        snap_idx = []
        snap_years = []
        for yy in years_to_show:
            idx = int(np.argmin(np.abs(yrs - int(yy))))
            snap_idx.append(idx)
            snap_years.append(int(yrs[idx]))

        fig2, axs2 = plt.subplots(len(snap_idx), 2, figsize=(14, 3.2 * len(snap_idx)), sharex=True)
        if len(snap_idx) == 1:
            axs2 = np.array([axs2])

        x = np.arange(len(sel))
        for r, (idx, yy) in enumerate(zip(snap_idx, snap_years)):
            mu_bar = mu_t[idx, sel]
            mu_ref = poe_mu[sel]
            sig_bar = sig_t[idx, sel]
            sig_ref = poe_sig[sel]

            axs2[r, 0].bar(x - 0.2, mu_bar, width=0.4, label=f"mu_t @ {yy}")
            axs2[r, 0].bar(x + 0.2, mu_ref, width=0.4, label="poe_mu")
            axs2[r, 0].set_title(f"Year {yy}: prior mean vs PoE pooled mean (top dims)")
            axs2[r, 0].set_ylabel("mu")
            axs2[r, 0].legend(fontsize=8)

            axs2[r, 1].bar(x - 0.2, sig_bar, width=0.4, label=f"sigma_t @ {yy}")
            axs2[r, 1].bar(x + 0.2, sig_ref, width=0.4, label="poe_sigma")
            axs2[r, 1].set_title(f"Year {yy}: prior std vs PoE pooled std (top dims)")
            axs2[r, 1].set_ylabel("sigma")
            axs2[r, 1].legend(fontsize=8)

        for c in range(2):
            axs2[-1, c].set_xlabel("latent dims (ranked)")
            axs2[-1, c].set_xticks(x)
            axs2[-1, c].set_xticklabels([f"z{j}" for j in sel], rotation=0)

        plt.suptitle(f"{bb}", fontsize = 14) 
        plt.tight_layout()
        plt.savefig(out_snap, bbox_inches="tight", dpi=180)
        plt.close(fig2)

        prec_mass_top = prec[:, sel].sum(axis=1)  # sum over dims => [T]
        df = pd.DataFrame({"year": yrs, "precision_mass_topdims_sum": prec_mass_top})
        df.to_csv(out_csv, index=False)

        return {"mu_and_precision_png": out_main, "year_snapshots_png": out_snap, "precision_mass_csv": out_csv}


    def run(self,
            which: str = "test",
            batch_idx: int = 0,
            time_idx: int = 0,
            sample_in_batch: int = 0,
            prior_which: Optional[str] = None,
            prefix: str = "pddeepsets",
            years: Optional[Sequence[int]] = None,
        ) -> Dict[str, str]:

        if self.model is None:
            self.load_model()

        dl = self._get_dl(which)
        x, pd_pts, pd_mask, pd_dim, y = self._nth_batch(dl, batch_idx)

        x = x.to(self.device, dtype=torch.float32)
        pd_pts = pd_pts.to(self.device, dtype=torch.float32)
        pd_mask = pd_mask.to(self.device, dtype=torch.float32)
        pd_dim = pd_dim.to(self.device, dtype=torch.long)

        # Full forward so the point-level conditioning attention beta is computed
        # and stashed on the conditioner. beta is the discriminative attention
        # (per-time queries attending over PD points); it is the object the
        # paper's attention figures should show, not the plain pooling alpha.
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x, pd_pts, pd_mask, pd_dim)

        # beta = getattr(self.model, "_last_point_attn", None)          # [B,T,P]
        # if beta is None:
        beta = getattr(self.model.tda_conditioner, "_last_point_attn", None)
        # plot code expects [B,T,P,1]
        w_beta = beta.unsqueeze(-1) if (beta is not None and beta.dim() == 3) else beta

        T = int(pd_pts.shape[1])
        # Guard the time axis against a mismatched years list: always align to T.
        if years is not None:
            years_arr = np.asarray(list(years))
        else:
            years_arr = np.asarray(list(self.args['n_years']))
        if len(years_arr) != T:
            # fall back to an index axis of the correct length rather than crash
            years_arr = np.arange(T)

        # Clamp the in-batch sample index to this batch's actual size.
        B = int(pd_pts.shape[0])
        b_sel = min(max(0, sample_in_batch), B - 1)
        
     
        paths = {
            "attn_mass_by_dim": os.path.join(self.folder, f"{prefix}_{which}_b{batch_idx}_s{b_sel}_condattn_mass_by_dim_over_years.png"),
            "attn_sharpness": os.path.join(self.folder, f"{prefix}_{which}_b{batch_idx}_s{b_sel}_condattn_sharpness_over_years.png"),
        }
        bb = self.args["backbone_model"]
        self.plot_attention_mass_by_dim_over_time(bb,
                                                  pd_mask,
                                                  pd_dim,
                                                  w_beta,
                                                  years_arr,
                                                  batch=b_sel,
                                                  out_path=paths["attn_mass_by_dim"])

        self.plot_attention_sharpness_over_time(bb,
                                                pd_mask,
                                                w_beta,
                                                years_arr,
                                                batch=b_sel,
                                                out_path=paths["attn_sharpness"])

        prior_paths = self.save_tda_prior(
                bb = bb,
                which=prior_which or which,
                batch_idx=batch_idx,
                sample_in_batch=b_sel,
                years_to_show=(1955, 1975, 1995, 2010),
                top_latent_dims=6,
                prefix="tdaprior",
                years=years_arr,
            )
        paths.update(prior_paths)

        return paths
