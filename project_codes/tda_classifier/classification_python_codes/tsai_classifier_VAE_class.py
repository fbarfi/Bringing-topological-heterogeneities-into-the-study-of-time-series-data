"""
TDA-conditioned sequence classifier with a topological variational prior.

This module implements the supervised classifier: a multivariate time-series
model that consumes, alongside the raw covariate panel, the persistence-diagram
(PD) stream produced by the topological pipeline, and uses that topological
information in two distinct ways.

Two topological pathways
------------------------
1. **Representation pathway (attention pooling and point-level conditioning).**
   For each timestep the PD is an unordered, variable-size set of points, each
   carrying a homology dimension. ``PDDeepSetsAttnMultiDim`` embeds every point
   together with a learned homology-dimension code, scores the points with a
   temperature-controlled attention head (alpha), and pools them into a
   fixed-width topological vector ``H_tda`` that enters the fusion stream and
   the prior. Separately, ``TDACrossAttentionConditioner`` conditions the
   backbone features: per-timestep queries built from ``[H_x_t || H_back0_t]``
   attend over the individual PD points of that year (attention beta), with
   keys/values computed from the per-point inputs by a dedicated feature map.
   The conditioner is read-only with respect to the PD points --- the backbone
   never writes back into the topological features.

2. **Regularization pathway (topological variational prior).** ``TDATimePrior``
   maps the per-timestep topological vector to a per-timestep Gaussian prior
   over the latent space, and fuses those per-time priors across the sequence
   with a Product-of-Experts (PoE) rule (``TDATimePrior.poe``). Because PoE sums
   precisions, timesteps whose topology yields a lower-variance prior
   contribute more to the pooled prior. A ``SeqVAE`` provides the posterior
   ``q(z|x)``; the KL term is taken against the pooled topological prior
   ``p(z|H)`` rather than a fixed ``N(0, I)``.

Fusion and prediction
----------------------
Three per-timestep streams are concatenated and passed through a bidirectional
fusion GRU: a light convolutional encoding of the raw covariates (``Conv1dEnc``),
the topological vector ``H_tda``, and the topology-conditioned backbone features.
Classification is read off the final timestep. A conditioner-only auxiliary
head (``cond_aux_head``) predicts the class from the pre-residual conditioner
output; its loss is weighted by the buffer ``lambda_cond`` (updated by the
``ConditionerDualAscent`` callback in ``models.py``). ``forward_with_aux``
exposes an auxiliary logit head on the fused representation, and
``forward_with_vae`` returns the posterior/prior statistics.

Backbone interface
------------------
Any tsai backbone is accepted by name (resolved via ``_get_backbone``) under a
uniform contract: instantiate with ``(c_in, c_out, seq_len)``, accept ``[B, F,
T]``, pool time internally, and return ``[B, c_out]``. The pooled output is
broadcast to ``[B, T, c_out]`` so the conditioner and fusion GRU receive
per-timestep inputs.

Tensor-shape conventions
------------------------
``B`` batch, ``T`` sequence length (``seq_len``), ``F`` covariates
(``feature_dim``), ``D`` VAE latent dim (``vae_latent_dim``). PD tensors are
``PD_pts`` ``[B, T, P, 2]`` (birth, death per point), ``PD_mask`` ``[B, T, P]``
(1 for real points, 0 for padding), and ``PD_dim`` ``[B, T, P]`` (homology
dimension per point, 0 reserved as the padding index for the embedding).
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tsai.all import *  

# ------------------------------ PD encoder  ------------------------------
class PDDeepSetsAttnMultiDim(nn.Module):
    """Attention-pooled Deep Sets encoder for homology-aware persistence diagrams.

    Maps the persistence diagram  of each timestep  --- an unordered, variable-length
    set of points spanning several homology dimensions --- to a single
    fixed-width topological vector, implementing the *representation* pathway of
    the paper. Each point is described by four scalars derived from its (birth,
    death) pair --- birth, death, persistence (death - birth), and midpoint ---
    concatenated with a learned embedding of its homology dimension, so that
    features from different dimensions remain distinguishable after pooling.

    Pooling follows the Deep Sets template with learned attention rather than a
    plain sum: a value network ``phi`` produces per-point features, an attention
    network ``psi`` produces per-point logits, and the diagram is summarised as
    the attention-weighted sum of ``phi`` features. A temperature ``tau``
    (optionally learned) controls how peaked the attention is --- small ``tau``
    concentrates on a few salient points, large ``tau`` spreads across many ---
    and padded points are masked out before the softmax so they receive zero
    weight. These pooling weights (alpha) are returned by ``attention_weights``;
    the point-level conditioning attention (beta) lives in
    ``TDACrossAttentionConditioner``.

    Args:
        out_dim: width of the pooled topological vector and of ``phi`` output.
        hidden: hidden width of the ``phi`` value MLP.
        attn_hidden: hidden width of the ``psi`` attention MLP.
        num_dims: number of homology dimensions (H_1..H_num_dims); the embedding
            table has ``num_dims + 1`` rows, with index 0 reserved as the
            padding slot (``padding_idx=0``).
        dim_emb: dimensionality of the learned homology-dimension embedding.
        tau: initial attention temperature.
        learn_tau: if True, ``tau`` is a trainable parameter; otherwise fixed.
        dropout: dropout applied inside ``phi``.

    Shapes:
        Inputs  --- ``PD_pts`` [B, T, P, 2], ``PD_mask`` [B, T, P],
                    ``PD_dim`` [B, T, P].
        Output  --- [B, T, out_dim].
    """
    def __init__(self, 
                 out_dim=64, 
                 hidden=128, 
                 attn_hidden=64, 
                 num_dims=4, 
                 dim_emb=8, 
                 tau=0.5, 
                 learn_tau=True, 
                 dropout=0.0):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(tau))) if learn_tau else float(tau)

        # Filled on each forward with the post-softmax pooling weights and the
        # point mask (for diagnostics).
        self._last_attn_w = None       # [B, T, P]  post-softmax weights
        self._last_attn_mask = None    # [B, T, P]  1 for real points, 0 for pad
        
        self.emb = nn.Embedding(num_embeddings=num_dims+1, embedding_dim=dim_emb, padding_idx=0)
        in_ch = 4 + dim_emb
        self.point_in_ch = in_ch   # width of per-point input x_in (for the conditioner)
        
        self.phi = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        
        self.psi = nn.Sequential(nn.Linear(in_ch, attn_hidden), 
                                 nn.GELU(), 
                                 nn.Linear(attn_hidden, 1))

        # Per-point buffers exposed for the point-level conditioner. We stash the
        # PRE-phi per-point input x_in (well-separated across points), NOT the
        # pooled-path features phi(x_in): phi is trained via the pooling path to
        # be point-invariant (uniform-mean pooling wants a stable mean), so it
        # collapses per-point variation. The conditioner must build its own
        # keys/values from x_in through a dedicated projection to get
        # discriminative point attention.
        self._last_point_in = None     # [B, T, P, in_ch]  pre-phi per-point input
        self._last_feats = None        # [B, T, P, out_dim] pooling-path features
        self._last_feat_mask = None    # [B, T, P]
        
        self.post = nn.Sequential(nn.LayerNorm(out_dim), 
                                  nn.Linear(out_dim, out_dim), 
                                  nn.GELU())

    @staticmethod
    def _prep_base(PD_pts: torch.Tensor, PD_mask: torch.Tensor = None):
        """Turn raw (birth, death) pairs into a 4-D per-point descriptor.

        Death is clamped to be at least birth (guarding against sub-diagonal
        noise), then each point is described by [birth, death, persistence,
        midpoint], where persistence = death - birth and midpoint =
        (birth + death) / 2. The four features are standardized within each
        diagram over the *real* (unpadded) points so that persistence---which
        for near-diagonal points is much smaller in raw magnitude than
        birth/death/midpoint---enters the scorer on a comparable scale.

        Args:
            PD_pts: [..., P, 2] birth/death coordinates.
            PD_mask: [..., P] 1 for real points, 0 for padding. If None, all
                points are treated as real.
        Returns:
            feat: [..., P, 4] standardized per-point features.
            pers: [..., P] raw (unstandardized) persistence, for the logit prior.
        """
        b = PD_pts[..., 0]
        d = torch.maximum(PD_pts[..., 1], b)
        l = d - b
        m = 0.5 * (b + d)
        feat = torch.stack([b, d, l, m], dim=-1)          # [..., P, 4]

        if PD_mask is None:
            msk = torch.ones_like(b)
        else:
            msk = (PD_mask > 0).to(feat.dtype)
        msk_e = msk.unsqueeze(-1)                          # [..., P, 1]

        # Masked mean/std over the point axis, using only real points, so padding
        # zeros do not corrupt the per-diagram statistics.
        cnt = msk_e.sum(dim=-2, keepdim=True).clamp_min(1.0)
        mean = (feat * msk_e).sum(dim=-2, keepdim=True) / cnt
        var = ((feat - mean).pow(2) * msk_e).sum(dim=-2, keepdim=True) / cnt
        std = var.clamp_min(1e-12).sqrt().clamp_min(1e-6)
        feat = (feat - mean) / std
        return feat, l

    def forward(self, PD_pts: torch.Tensor, PD_mask: torch.Tensor, PD_dim: torch.Tensor) -> torch.Tensor:
        """Encode and attention-pool one PD set per timestep.

        Computes per-point value features (``phi``) and pooling logits (``psi``),
        forms a plain masked, tempered attention pool alpha over points, and
        returns the pooled per-time topology vector. The pooled vector feeds the
        fusion concatenation and the topological prior. Point selection for
        conditioning is handled by the point-level cross-attention conditioner,
        which reads the per-point input stashed on ``self._last_point_in``
        (with the mask on ``self._last_feat_mask``).

        Args:
            PD_pts: [B, T, P, 2] birth/death coordinates.
            PD_mask: [B, T, P] with 1 for real points and 0 for padding.
            PD_dim: [B, T, P] homology dimension per point (0 = padding index).
        Returns:
            [B, T, out_dim] pooled topological vector per timestep.
        """
        base, pers = self._prep_base(PD_pts, PD_mask)
        dim_e = self.emb(PD_dim.clamp_min(0))
        x_in  = torch.cat([base, dim_e], dim=-1)
        feats  = self.phi(x_in)                            # [B, T, P, out_dim]
        logits = self.psi(x_in).squeeze(-1)                # [B, T, P]

        tau = self.tau.clamp_min(1e-3) if isinstance(self.tau, torch.Tensor) else max(1e-3, float(self.tau))
        logits = logits / tau
        very_neg = torch.finfo(feats.dtype).min / 4
        logits = torch.where(PD_mask > 0, logits, torch.full_like(logits, very_neg))
        w = F.softmax(logits, dim=2)                       # [B, T, P] pooling attention alpha

        # Expose per-point pre-phi input (well-separated) for the conditioner,
        # plus the pooling features and mask.
        self._last_point_in = x_in                         # [B, T, P, in_ch]
        self._last_feats = feats                           # [B, T, P, out_dim]
        self._last_feat_mask = (PD_mask > 0).to(feats.dtype)
        self._last_attn_w = w                              # alpha (pooling), for diagnostics
        self._last_attn_mask = (PD_mask > 0).to(w.dtype)

        w = w.unsqueeze(-1)
        agg = (feats * w).sum(dim=2)
        return self.post(agg)

    @torch.no_grad()
    def attention_weights(self, PD_pts: torch.Tensor, PD_mask: torch.Tensor, PD_dim: torch.Tensor) -> torch.Tensor:
        """Return the post-softmax attention weights [B, T, P] for a PD batch.

        Reproduces exactly the pooling attention (alpha) computed inside
        ``forward`` (temperature + masking) so diagnostics read the same pooling
        weights the model uses. Note: this is the *pooling* attention alpha that
        feeds the fusion/prior, not the point-level conditioning attention beta
        (which lives in the conditioner and is the discriminative one).
        """
        base, _pers = self._prep_base(PD_pts, PD_mask)
        dim_e = self.emb(PD_dim.clamp_min(0))
        x_in  = torch.cat([base, dim_e], dim=-1)
        logits = self.psi(x_in).squeeze(-1)

        tau = self.tau.clamp_min(1e-3) if isinstance(self.tau, torch.Tensor) else max(1e-3, float(self.tau))
        logits = logits / tau
        very_neg = torch.finfo(logits.dtype).min / 4
        logits = torch.where(PD_mask > 0, logits, torch.full_like(logits, very_neg))
        return F.softmax(logits, dim=2)

# ------------------------------ VAE  ------------------------------
class SeqVAE(nn.Module):
    """Sequence variational autoencoder supplying the posterior q(z|x).

    A 1-D convolutional encoder maps the covariate panel [B, T, F] to a global
    latent posterior (``mu``, ``logvar``) via adaptive average pooling over
    time; a linear decoder reconstructs the full [B, T, F] panel from a latent
    sample. Its role in the wider model is to provide the posterior that is
    regularized against the *topological* prior produced by ``TDATimePrior``,
    rather than against a fixed ``N(0, I)``. The static
    ``loss_with_conditional_prior`` computes reconstruction + beta * KL, with the
    KL taken between q(z|x) and the supplied prior --- the single term the
    paper's VAE ablation toggles.

    Args:
        c_in: number of input covariates F.
        seq_len: sequence length T (fixes the decoder output width).
        latent_dim: latent dimensionality D.
        h: hidden width of encoder/decoder.

    Note:
        The classifier calls ``encode``/``reparam``/``decode`` directly;
        ``forward`` runs the same three steps for standalone use.
    """
    def __init__(self, c_in, seq_len, latent_dim=128, h=128):  
        super().__init__()
        self.seq_len, self.c_in, self.latent_dim = int(seq_len), int(c_in), int(latent_dim)
        self.enc = nn.Sequential(
            nn.Conv1d(c_in,h,5,padding=2), 
            nn.GELU(),
            nn.Conv1d(h,h,3,padding=1), 
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        
        self.mu = nn.LazyLinear(latent_dim)
        self.logvar = nn.LazyLinear(latent_dim)

        self.dec = nn.Sequential(nn.Linear(latent_dim,h), 
                                 nn.GELU(), 
                                 nn.Linear(h, self.c_in*self.seq_len))

    def encode(self, x): 
        h = self.enc(x.transpose(1,2)).squeeze(-1)
        return self.mu(h), self.logvar(h)
    @staticmethod
    def reparam(mu, logvar): 
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std
    
    def decode(self, z): 
        rec = self.dec(z)
        return rec.view(z.size(0), self.c_in, self.seq_len).transpose(1,2)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)        # [B, D], [B, D]
        z = self.reparam(mu, logvar)       # [B, D]
        recon = self.decode(z)             # [B, T, F]
        return z, mu, logvar, recon

    @staticmethod
    def loss_with_conditional_prior(
        x: torch.Tensor,
        recon: torch.Tensor,
        q_mu: torch.Tensor, q_logvar: torch.Tensor,
        p_mu: torch.Tensor, p_logvar: torch.Tensor,
        beta: float = 1.0,
        recon_mode: str = "mse",
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO-style loss with an arbitrary (here topological) Gaussian prior.

        Computes reconstruction loss plus ``beta`` times the analytic KL between
        two diagonal Gaussians, KL(q(z|x) || p(z|H)), using the posterior
        (``q_mu``, ``q_logvar``) and the pooled prior (``p_mu``, ``p_logvar``).
        Setting the prior to (0, 0) recovers the standard N(0, I) baseline used
        in the ablation.

        Args:
            x, recon: ground-truth and reconstructed panels [B, T, F].
            q_mu, q_logvar: posterior parameters [B, D].
            p_mu, p_logvar: prior parameters [B, D].
            beta: KL weight (annealed via the classifier's learnable ``beta``).
            recon_mode: 'mse' or 'mae'.
            eps: numerical floor on the prior variance.
        Returns:
            (total_loss, recon_loss, kl), each a scalar.
        """
        if recon_mode == "mse":
            recon_loss = F.mse_loss(recon, x, reduction="mean")
        elif recon_mode == "mae":
            recon_loss = F.l1_loss(recon, x, reduction="mean")
        else:
            raise ValueError("Unsupported recon_mode; expected 'mse' or 'mae'.")
        q_var = q_logvar.exp()
        p_var = p_logvar.exp()
        kl = 0.5 * torch.mean(
            (q_var + (q_mu - p_mu).pow(2)) / (p_var + eps) - 1.0 + (p_logvar - q_logvar)
        )
        return recon_loss + beta * kl, recon_loss, kl

# ------------------------------ TDA prior  ------------------------------
class TDATimePrior(nn.Module):
    """Maps the topological vector to a per-time Gaussian prior, then PoE-fuses it.

    Implements the *regularization* pathway. An MLP turns each timestep's
    topological vector ``H_tda`` into the parameters (``mu_t``, ``logvar_t``) of
    a diagonal Gaussian prior over the latent space; ``logvar`` is clamped to
    [-8, 8] for numerical stability. The static ``poe`` then fuses the per-time
    priors across the sequence with a Product-of-Experts rule.

    Args:
        tda_dim: width of the incoming topological vector.
        latent_dim: latent dimensionality D (the prior's dimension).
        hidden: hidden width of the mapping MLP.

    Shapes:
        ``forward``: [B, T, tda_dim] -> (mu [B, T, D], logvar [B, T, D]).
    """
    def __init__(self, tda_dim, latent_dim, hidden=128):
        super().__init__() 
        self.net = nn.Sequential(nn.Linear(tda_dim, hidden), 
                                 nn.GELU(), 
                                 nn.Linear(hidden, 2*latent_dim))
    def forward(self, H): 
        h = self.net(H)
        mu, lv = torch.chunk(h, 2, dim=-1)
        return mu, lv.clamp(-8,8)
    
    @staticmethod
    def poe(mu_t: torch.Tensor, logvar_t: torch.Tensor, eps: float = 1e-6):
        """Product-of-Experts fusion of per-time Gaussian priors.

        Treats each timestep as an independent Gaussian expert and multiplies
        them: precisions (inverse variances) sum, and the pooled mean is the
        precision-weighted average of the per-time means. Timesteps whose
        topology yields lower variance (higher precision) therefore dominate the
        pooled prior --- the "reliability weighting" described in the paper.

        Args:
            mu_t: [B, T, D] per-time prior means.
            logvar_t: [B, T, D] per-time prior log-variances.
            eps: numerical floor on variances/precision sum.
        Returns:
            (mu [B, D], logvar [B, D]) of the pooled prior.
        """
        var_t = logvar_t.exp() + eps
        prec_t = 1.0 / var_t
        prec_sum = prec_t.sum(dim=1) + eps
        mu = (mu_t * prec_t).sum(dim=1) / prec_sum
        logvar = torch.log(1.0 / prec_sum)
        return mu, logvar

# ------------------------------ Light temporal encoder  ------------------------------
class Conv1dEnc(nn.Module):
    """Light two-layer 1-D conv encoder producing per-timestep covariate features.

    A padding-preserving temporal encoder over the raw covariates: it keeps the
    sequence length T (no pooling) so its output can be fused per-timestep with
    the topological and backbone streams. Transposes to channels-first for the
    convolutions and back to [B, T, d] on output.

    Args:
        in_ch: number of input covariates F.
        d: output channel width.
    Shapes:
        [B, T, F] -> [B, T, d].
    """
    def __init__(self, in_ch, d): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,d,3,padding=1), 
            nn.GELU(), 
            nn.BatchNorm1d(d),
            nn.Conv1d(d,d,3,padding=1), 
            nn.GELU(), 
            nn.BatchNorm1d(d))
    def forward(self,x): 
        return self.net(x.transpose(1,2)).transpose(1,2)


# ------------------------------ TDA Cross-Attention Conditioner ------
class TDACrossAttentionConditioner(nn.Module):
    """Per-time queries attend over PD *points* to condition the backbone.

    This is the point-level conditioner. Unlike a time-level conditioner that
    attends over pooled per-year topology vectors, here the query at each time
    step attends over the individual persistence-diagram points of that year,
    selecting which points are relevant given the current temporal context. The
    per-point attention beta_{t,p} is thus driven directly by the classification
    gradient through the queries---realizing discriminative point selection
    structurally, rather than coaxing a separate pooling attention to be
    selective.

    Design:
      - Query q_t is built from the per-time context [H_x_t || H_back0_t], so it
        is genuinely time-varying for every backbone (the light-conv stream H_x
        always varies over t, even when the backbone's broadcast features do not).
      - Keys/Values come from c_{t,p} = phi_cond(e_{t,p}), a dedicated feature
        map applied to the per-point PD encoder input e_{t,p} (masked), so the
        conditioner sees points, not their pooled summary.
      - Attention is over points within each (b, t); padded points are masked to
        -inf before the softmax.
      - Output folds into the backbone via a residual, LayerNorm'd projection,
        preserving the read-only role of the topology stream (points are read,
        never modified).

    Output = LayerNorm(H_back0 + Dropout(W_O * context)),
      context_t = sum_p beta_{t,p} * (W_V c_{t,p}).
    """
    def __init__(self, backbone_dim, tda_dim, query_ctx_dim, point_in_ch, n_heads=4, dropout=0.1, cond_tau_init=0.05):
        super().__init__()
        assert backbone_dim % n_heads == 0, \
            f"backbone_dim ({backbone_dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k = backbone_dim // n_heads

        # Query from per-time context [H_x_t || H_back0_t] -> backbone_dim
        self.W_Q = nn.Linear(query_ctx_dim, backbone_dim, bias=False)

        # Dedicated per-point feature extractor for the conditioner. Crucially
        # SEPARATE from the PD encoder's phi: phi is trained through the pooling
        # path to be point-invariant, which would starve the conditioner of
        # per-point contrast. phi_cond maps the pre-phi per-point input directly
        # to keys/values and is trained ONLY through the classification gradient
        # via beta, so it is free to become point-discriminative.
        self.phi_cond = nn.Sequential(
            nn.Linear(point_in_ch, backbone_dim), nn.GELU(),
            nn.Linear(backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
        )
        self.W_K = nn.Linear(backbone_dim, backbone_dim, bias=False)
        self.W_V = nn.Linear(backbone_dim, backbone_dim, bias=False)
        self.W_O = nn.Linear(backbone_dim, backbone_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(backbone_dim)

        # Learnable attention temperature. Softmax over hundreds of PD points
        # needs large logit gaps to concentrate; the standard 1/sqrt(d_k) scaling
        # (tuned for a few tokens) crushes contrast here. We parameterize the
        # temperature as exp(log_tau) so it stays positive, initialize it small
        # (sharp), and let the conditioner-aux gradient tune it -- the aux head
        # rewards concentration, so tau has both a reason and a path to shrink.
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(cond_tau_init))))

        self._last_point_attn = None
        self._last_point_mask = None

    def forward(self, H_back0, point_in, pd_mask, query_ctx):
        """
        H_back0:   [B, T, backbone_dim]   backbone per-time features (for residual)
        point_in:  [B, T, P, point_in_ch] pre-phi per-point PD input (keys/values source)
        pd_mask:   [B, T, P]              1 for real points, 0 for padding
        query_ctx: [B, T, query_ctx_dim]  per-time query context [H_x || H_back0]
        Returns:   [B, T, backbone_dim]   point-conditioned backbone features
        """
        B, T, P, _ = point_in.shape
        nh, dk = self.n_heads, self.d_k

        # Dedicated per-point features for conditioning (point-discriminative).
        cfeat = self.phi_cond(point_in)                                    # [B,T,P,backbone_dim]

        # Queries: one per (b, t). [B, nh, T, 1, dk]
        Q = self.W_Q(query_ctx).view(B, T, nh, dk).transpose(1, 2)          # [B,nh,T,dk]
        Q = Q.unsqueeze(3)                                                  # [B,nh,T,1,dk]

        # Keys/Values from per-point conditioning features. [B, nh, T, P, dk]
        K = self.W_K(cfeat).view(B, T, P, nh, dk).permute(0, 3, 1, 2, 4)    # [B,nh,T,P,dk]
        V = self.W_V(cfeat).view(B, T, P, nh, dk).permute(0, 3, 1, 2, 4)    # [B,nh,T,P,dk]

        # Scores over points, scaled by 1/sqrt(dk) and a learnable temperature.
        # Small tau => sharper attention (needed to concentrate over many points).
        tau = self.log_tau.exp().clamp_min(1e-3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / ((dk ** 0.5) * tau)
        scores = scores.squeeze(3)

        m = (pd_mask > 0).view(B, 1, T, P)
        very_neg = torch.finfo(scores.dtype).min / 4
        scores = torch.where(m, scores, torch.full_like(scores, very_neg))

        beta = F.softmax(scores, dim=-1)                                   # [B,nh,T,P]
        beta = self.dropout(beta)

        context = torch.matmul(beta.unsqueeze(3), V).squeeze(3)            # [B,nh,T,dk]
        context = context.transpose(1, 2).contiguous().view(B, T, -1)     # [B,T,backbone_dim]

        self._last_point_attn = beta.mean(dim=1).detach()                 # [B,T,P] head-averaged
        self._last_point_attn_heads = beta.detach()                       # [B,nh,T,P] per-head
        self._last_point_mask = (pd_mask > 0).to(beta.dtype).detach()

        out = self.W_O(context)
        # Expose the pure conditioner contribution (pre-residual) so an auxiliary
        # head can be driven by it alone. This makes beta causally necessary:
        # the aux head can only predict through the conditioner output, with no
        # backbone residual to bypass it, so gradient flows into beta/phi_cond.
        self._last_contrib = out                                          # [B,T,backbone_dim]
        return self.norm(H_back0 + self.dropout(out))


# ------------------------------ Backbone helpers -----------------------
def _get_backbone(name: str):
    bb = globals().get(name, None)
    if bb is None or not callable(bb):
        raise ValueError(f"Backbone model '{name}' not found or not callable.")
    return bb



class TSAIClassifier_VAE(nn.Module):
    """TDA-conditioned sequence classifier with a topological VAE prior.

    Ties the two topological pathways together around an arbitrary tsai backbone
    (see the module docstring for the overall design). On each ``forward`` the
    model (1) encodes the PD stream into ``H_tda`` and a pooled topological prior
    ``p(z|H)``; (2) runs the ``SeqVAE`` to obtain the posterior ``q(z|x)`` and an
    optional reconstruction; (3) computes three per-timestep streams --- a light
    conv encoding of the covariates, ``H_tda``, and backbone features; (4)
    conditions the backbone features on the individual PD points through the
    read-only point-level cross-attention conditioner; and
    (5) fuses all three through a bidirectional GRU and classifies the final
    timestep. Posterior, reconstruction, and prior statistics are cached on
    ``self._last_*`` each forward so the external VAE loss can be evaluated
    without a second pass.

    Args (via the ``args`` dict):
        feature_dim: number of covariates F (required).
        seq_len: sequence length T (required).
        hidden_size: GRU/head hidden width, and backbone ``c_out`` (required).
        num_classes: number of output classes (required).
        num_layers: GRU depth (required).
        vae_latent_dim: latent dim D (default 128).
        pd_out_dim: width of ``H_tda`` (default 128).
        backbone_model: name of a tsai backbone visible in this module's globals
            (required; resolved by ``_get_backbone``).
        beta_min/beta_max/beta_init: bounds and initial value for the learnable
            KL weight ``beta`` (sigmoid-parameterized; see the ``beta`` property).
        pd_hidden, pd_attn_hidden, num_dims, dim_emb, pd_tau, pd_learn_tau,
            pd_dropout: forwarded to ``PDDeepSetsAttnMultiDim``.
        tda_cond_heads, tda_cond_dropout: heads/dropout for the cross-attention
            conditioner (heads reduced automatically to divide the backbone dim).

    Forward signature:
        ``forward(x, PD_pts, PD_mask, PD_dim) -> logits [B, num_classes]``,
        with ``x`` [B, T, F] and PD tensors as in the module docstring.
    """
    def __init__(self, args: Dict):
        super().__init__()
        self.feature_dim  = int(args['feature_dim'])
        self.seq_len      = int(args['seq_len'])
        self.hidden_size  = int(args['hidden_size'])
        self.num_classes  = int(args['num_classes'])
        self.num_layers   = int(args['num_layers'])
        self.vae_latent_dim = int(args.get('vae_latent_dim', 128))
        pd_out = int(args.get('pd_out_dim', 128))

        # ---- Learnable beta for VAE ----
        beta_min = float(args.get("beta_min", 0.0))
        beta_max = float(args.get("beta_max", 5.0))
        beta_init = float(args.get("beta_init", 1.0))
        self._beta_min, self._beta_max = beta_min, beta_max
        p = (min(max(beta_init, beta_min + 1e-6), beta_max - 1e-6) - beta_min) / (beta_max - beta_min)
        init_logit = torch.log(torch.tensor(p) / (1 - torch.tensor(p)))
        self.beta_logit = nn.Parameter(init_logit.view(1))

        # Stored from args; not used by the training loss
        # (VAECompositeLossUsingModelBeta in models.py).
        self.attn_entropy_lambda = float(args.get('attn_entropy_lambda', 0.0))

        # ---- VAE bookkeeping ----
        self.collect_vae = True
        self._q_mu = self._q_lv = self._recon = None
        self._prior_mu = self._prior_lv = None
        self._last_x = None
        self._last_mu = None
        self._last_logvar = None
        self._last_recon = None
        self._last_prior_mu = None
        self._last_prior_logvar = None

        # ---- Backbone setup ----
        # All tsai backbones share the same interface: instantiate with
        # (c_in, c_out, seq_len), accept [B, F, T], pool time internally,
        # and return [B, c_out]. Time-pooled output is broadcast to
        # [B, T, c_out] for the per-timestep conditioner / fusion GRU.
        self.backbone_name = str(args['backbone_model'])
        backbone_cls = _get_backbone(self.backbone_name)
        self.backbone = backbone_cls(
            c_in=self.feature_dim,
            c_out=self.hidden_size,
            seq_len=self.seq_len,
        )
        backbone_out_dim = self.hidden_size

        # ---- PD encoder ----
        self.pd_enc = PDDeepSetsAttnMultiDim(
            out_dim=pd_out, 
            hidden=int(args.get('pd_hidden', 128)),
            attn_hidden=int(args.get('pd_attn_hidden', 128)),
            num_dims=int(args.get('num_dims', 4)), 
            dim_emb=int(args.get('dim_emb', 8)),
            tau=float(args.get('pd_tau', 0.5)), 
            learn_tau=bool(args.get('pd_learn_tau', True)),
            dropout=float(args.get('pd_dropout', 0.0)),
        )

        # ---- VAE  ----
        self.vae = SeqVAE(c_in=self.feature_dim, 
                          seq_len=self.seq_len, 
                          latent_dim=self.vae_latent_dim, 
                          h=max(128, self.feature_dim * 2))

        # ---- TDA prior  ----
        self.prior = TDATimePrior(tda_dim=pd_out, 
                                  latent_dim=self.vae_latent_dim, 
                                  hidden=max(128, self.vae_latent_dim * 2))

        # ---- Light temporal encoder ----
        hx_dim = min(128, self.hidden_size)
        self.x_temporal = Conv1dEnc(in_ch=self.feature_dim, d=hx_dim)

        # ---- TDA point-level cross-attention conditioner ----
        # Per-time queries [H_x_t || H_back0_t] attend over the PD *points* of
        # that year (keys/values = per-point features). The point attention beta
        # is driven by the classification gradient through the queries, so point
        # selection is discriminative and (via H_x) genuinely time-varying for
        # every backbone. TDA points are read-only.
        n_cond_heads = int(args.get('tda_cond_heads', 4))
        cond_dropout = float(args.get('tda_cond_dropout', 0.1))

        # Ensure backbone_out_dim is divisible by n_cond_heads
        while backbone_out_dim % n_cond_heads != 0 and n_cond_heads > 1:
            n_cond_heads -= 1

        self.tda_conditioner = TDACrossAttentionConditioner(
            backbone_dim=backbone_out_dim,
            tda_dim=pd_out,
            query_ctx_dim=hx_dim + backbone_out_dim,   # [H_x || H_back0] per time
            point_in_ch=self.pd_enc.point_in_ch,       # pre-phi per-point input width
            n_heads=n_cond_heads,
            dropout=cond_dropout,
            cond_tau_init=float(args.get("cond_tau_init", 0.05)),
        )

        # ---- Fusion GRU ----
        # All three inputs are per-timestep:
        #   Hx:     [B, T, hx_dim]         from Conv1dEnc
        #   H_tda:  [B, T, pd_out]         from PD encoder
        #   H_back: [B, T, backbone_out]   from conditioned backbone
        fuse_in = hx_dim + pd_out + backbone_out_dim

        self.rnn = nn.GRU(input_size=fuse_in, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          bidirectional=True, batch_first=True)
        
        self.rnn_do = nn.Dropout(0.30)

        # ---- Classification head ----
        self.head = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_size), 
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(self.hidden_size, self.num_classes),
        )

        # ---- Auxiliary head (for forward_with_aux) ----
        self.head_logit = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, fuse_in // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fuse_in // 2, self.num_classes),
        )

        # ---- Conditioner-only auxiliary head ----
        # Predicts the class from the PURE conditioner contribution alone (the
        # pre-residual W_O*context, pooled over time). Because this head has no
        # backbone-residual bypass, its loss forces the point attention beta to
        # carry discriminative signal -- making beta causally necessary instead
        # of an ignorable add-on. Weighted by cond_aux_weight in the objective.
        self.cond_aux_head = nn.Sequential(
            nn.LayerNorm(backbone_out_dim),
            nn.Linear(backbone_out_dim, backbone_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(backbone_out_dim // 2, self.num_classes),
        )

        # Conditioner ablation switch (used by the dual-ascent callback to run a
        # conditioner-off validation pass for the on-the-fly reference).
        self.conditioner_enabled = True

        # Learned dual multiplier for the conditioner-aux loss. NOT a hand-set
        # weight: updated by GECO-style dual ascent in a callback so the aux term
        # is weighted only as much as keeps the conditioner from worsening
        # held-out loss versus its ablation. A buffer (updated by the callback,
        # not by gradient); grows while the conditioner helps validation, shrinks
        # when it hurts.
        self.register_buffer("lambda_cond", torch.tensor(float(args.get('cond_aux_lambda_init', 0.1))))
        self.cond_aux_lambda_max = float(args.get('cond_aux_lambda_max', 2.0))

    @property
    def beta(self):
        """Learnable KL weight, squashed into [beta_min, beta_max] via sigmoid."""
        s = torch.sigmoid(self.beta_logit)
        return self._beta_min + (self._beta_max - self._beta_min) * s

    def _backbone_features(self, x):
        """Extract per-timestep backbone features.

        All tsai backbones expect [B, F, T], pool time internally, and
        return [B, D]. We broadcast that pooled vector to [B, T, D] so the
        TDA cross-attention conditioner and fusion GRU receive per-timestep
        inputs.

        Args:
            x: [B, T, F] raw covariates
        Returns:
            [B, T, backbone_out_dim] per-timestep features
        """
        x_tr = x.float().transpose(1, 2)                              # [B, F, T]
        x_bk = self.backbone(x_tr)                                    # [B, D] (pooled)
        T = x.shape[1]
        return x_bk.unsqueeze(1).expand(-1, T, -1)                    # [B, T, D] (broadcast)

    def forward(self, x, PD_pts, PD_mask, PD_dim):
        """Run both topological pathways, fuse the streams, and classify.

        Steps: (1) encode the PD stream to ``H_tda`` and PoE-pool the per-time
        topological prior; (2) VAE-encode ``x`` to the posterior and (optionally)
        reconstruct; (3) build the light conv and backbone per-timestep streams;
        (4) condition the backbone on the PD points via read-only point-level
        cross-attention;
        (5) concatenate the three streams and run the bidirectional fusion GRU;
        (6) classify from the last timestep. Posterior/prior/recon tensors are
        stashed on ``self._last_*`` and ``self._prior_*`` for the external loss.

        Args:
            x: [B, T, F] covariate panel.
            PD_pts, PD_mask, PD_dim: PD stream (see module docstring).
        Returns:
            logits [B, num_classes] from the final timestep.
        """
        # ----- 1) TDA -> topology vectors + prior (pooled) -----
        H_tda = self.pd_enc(PD_pts.float(), PD_mask.float(), PD_dim.long())    # [B, T, pd_out]

        # Surface the PD-encoder pooling weights (alpha) for diagnostics.
        self._last_attn_w = getattr(self.pd_enc, "_last_attn_w", None)
        self._last_attn_mask = getattr(self.pd_enc, "_last_attn_mask", None)
        
        mu_t, lv_t = self.prior(H_tda)                                         # [B, T, D]
        prior_mu, prior_lv = TDATimePrior.poe(mu_t, lv_t)                      # [B, D]
        self._prior_mu, self._prior_lv = prior_mu, prior_lv

        # ----- 2) VAE encode/decode (for KL + recon) -----
        mu, lv = self.vae.encode(x.float())                                    # [B, D], [B, D]
        z = self.vae.reparam(mu, lv)
        recon = self.vae.decode(z) if self.collect_vae else None

        self._q_mu, self._q_lv = mu, lv
        self._recon = recon
        self._last_x = x
        self._last_mu = mu
        self._last_logvar = lv
        self._last_recon = recon
        self._last_prior_mu = prior_mu
        self._last_prior_logvar = prior_lv

        # ----- 3) Temporal encoders (all per-timestep) -----
        
        # 3a) Light conv encoder over raw covariates
        Hx = self.x_temporal(x.float())                                        # [B, T, hx_dim]
        Hx = F.dropout(Hx, p=0.10, training=self.training)

        # 3b) Backbone features: pooled [B, D] broadcast to [B, T, D]
        H_back_raw = self._backbone_features(x)                                # [B, T, backbone_out_dim]

        # ----- 4) TDA conditions backbone via POINT-level cross-attention -----
        #     Per-time queries [H_x_t || H_back0_t] attend over the PD points of
        #     that year (keys/values from phi_cond on the PD encoder's per-point
        #     input).
        #     The point attention beta is discriminative (driven by the
        #     classification gradient through the queries) and time-varying (via
        #     H_x). TDA points are read-only.
        point_in = self.pd_enc._last_point_in                                 # [B, T, P, in_ch]
        feat_mask = self.pd_enc._last_feat_mask                                # [B, T, P]
        query_ctx = torch.cat([Hx, H_back_raw], dim=-1)                        # [B, T, hx+bb]
        if getattr(self, "conditioner_enabled", True):
            H_back = self.tda_conditioner(H_back_raw, point_in, feat_mask, query_ctx)  # [B, T, backbone_out_dim]
        else:
            # Ablated pass: skip the conditioner contribution entirely so
            # H_back == H_back0. Used by the dual-ascent callback to measure the
            # conditioner's marginal effect on held-out loss (conditioner off).
            H_back = H_back_raw

        # Surface the point-level conditioning attention (the discriminative one)
        # for diagnostics/plots.
        self._last_point_attn = getattr(self.tda_conditioner, "_last_point_attn", None)
        self._last_point_attn_heads = getattr(self.tda_conditioner, "_last_point_attn_heads", None)
        self._last_point_mask = getattr(self.tda_conditioner, "_last_point_mask", None)

        # Conditioner-only auxiliary logits: predict from the pure conditioner
        # contribution (pre-residual), pooled over time. Stashed for the loss to
        # add a term that makes beta causally required.
        contrib = getattr(self.tda_conditioner, "_last_contrib", None)         # [B, T, bb]
        if contrib is not None:
            self._cond_aux_logits = self.cond_aux_head(contrib.mean(dim=1))    # [B, C]
        else:
            self._cond_aux_logits = None

        # ----- 5) Fuse + GRU -----
        fused = torch.cat([Hx, H_tda, H_back], dim=-1)                        # [B, T, fuse_in]
        fused = F.dropout(fused, p=0.10, training=self.training)

        Hf, _ = self.rnn(fused)                                                # [B, T, 2*hidden]
        Hf = self.rnn_do(Hf)

        # ----- 6) Class head on last step -----
        h_last = Hf[:, -1, :]                                                  # [B, 2*hidden]
        logits = self.head(h_last)                                             # [B, C]
        return logits
    
    def forward_with_aux(self, x, PD_pts, PD_mask, PD_dim):
        """Main logits plus an auxiliary logit head on the fused representation.

        Runs the standard ``forward`` for the primary logits, then recomputes the
        fused per-timestep representation and applies ``head_logit`` to its final
        timestep. The auxiliary head (a shallower classifier on the pre-GRU
        fusion) supports deep-supervision / auxiliary-loss training.

        Returns:
            (logits, aux_logits), each [B, num_classes].
        """
        logits = self.forward(x, PD_pts, PD_mask, PD_dim)
        
        # Recompute fused representation for aux head
        H_tda = self.pd_enc(PD_pts.float(), PD_mask.float(), PD_dim.long())
        Hx = self.x_temporal(x.float())
        H_back_raw = self._backbone_features(x)
        point_in = self.pd_enc._last_point_in
        feat_mask = self.pd_enc._last_feat_mask
        query_ctx = torch.cat([Hx, H_back_raw], dim=-1)
        H_back = self.tda_conditioner(H_back_raw, point_in, feat_mask, query_ctx)
        fused = torch.cat([Hx, H_tda, H_back], dim=-1)
        
        aux_logits = self.head_logit(fused[:, -1, :])
        return (logits, aux_logits)

    def forward_with_vae(self, x, PD_pts, PD_mask, PD_dim):
        """
        Runs the standard forward (which fills _last_* buffers) and
        returns logits plus the VAE/posterior + prior stats for inspection.
        """
        logits = self.forward(x, PD_pts, PD_mask, PD_dim)

        return (
            logits,                 # [B, C]
            self._last_mu,          # q(z|x) mean     [B, D]
            self._last_logvar,      # q(z|x) logvar   [B, D]
            self._last_recon,       # VAE recon x̂     [B, T, F] (or None)
            self._prior_mu,         # pooled p(z|H) μ [B, D]
            self._prior_lv,         # pooled p(z|H) lv[B, D]
        )
