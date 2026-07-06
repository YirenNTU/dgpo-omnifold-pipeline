"""Object-token bottleneck autoencoder: event CLS token + per-object tokens + neutrinos.

The **single** latent-constraint model for DGPO. Inputs are the frozen EveNet
event representation -- the pooled **event CLS token** (``event_token``) plus the
**full set of per-object ObjectEncoder tokens** (``object_token``: ``(B, P, D)``,
both from ``preprocessing/augment_event_token.py --object-tokens``) -- together
with the two-neutrino kinematics. The reconstruction target is the **original
pretrain-model event token** and the **neutrino kinematics**.

Design highlights:

1. **Rich input conditioning.** The neutrinos share one self-attention stack with
   the event CLS token and all object tokens, so the ν↔object (e.g. ν↔lepton)
   correlation is available *before* the bottleneck pools ``z`` -- no assignment /
   no physics label: attention picks the relevant object itself. Objects enter as
   input only; they are never a reconstruction target.

2. **No loss weight (hyperparameter-free).** Both targets are already standardized to
   ~unit variance -- the neutrino Normalizer (z-score + inv-CDF phi) and the event-token
   standardization -- and each MSE is mean-reduced over its dims, so ``mse_nu`` and
   ``mse_token`` are the same dimensionless unit ("mean per-dim normalized residual").
   The loss is therefore just their SUM (implicit weight 1, principled -- not tuned), and
   a task that converges naturally fades out (``grad -> 0``) instead of being amplified.

3. **DGPO drop-in conventions.** The non-conditional decoder makes ``recon`` a
   faithful readout of what the bottleneck ``z`` encodes; ``z`` is the deployment
   latent for the sliced-Wasserstein constraint. Neutrino inputs are **physical**
   and normalized *inside* the model with a differentiable mirror of the EveNet
   ``Normalizer``, so gradients flow from ``z_pred`` back to the DGPO prediction
   path even with a frozen encoder. ``encode_latent`` is deliberately NOT
   ``@torch.no_grad()``. Build from the **same normalization.pt** as the policy.

Checkpoint I/O (:func:`save_checkpoint` / :func:`load_checkpoint` /
:func:`parse_lc_resume_from_checkpoint`) lives here too, making this module the
one canonical home of the latent-constraint model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from evenet.network.body.normalizer import Normalizer
from RL.DGPO_neutrino.latent_constraint.normalizers import load_normalizers_from_pt

_NU_KIN_SLICE_7 = slice(1, 4)
MODEL_TYPE = "object_token_bottleneck_ae"
_CHECKPOINT_VERSION = 1


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


class ObjectTokenBottleneckAutoencoder(nn.Module):
    """[z-cls, event-token, object-tokens..., nu, antinu] self-attn -> bottleneck z."""

    def __init__(
        self,
        normalization_file: str | None = None,
        *,
        normalizers: tuple[Normalizer, Normalizer, Normalizer] | None = None,
        token_dim: int = 256,
        nu_kin_dim: int = 3,
        d_model: int | None = None,
        latent_dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        cartesian: bool = False,
        phi_index: int | None = 2,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if normalizers is None and normalization_file is None:
            raise ValueError("provide either normalization_file or normalizers")

        self.cartesian = bool(cartesian)
        self.invisible_key = "x_invisible_cartesian" if self.cartesian else "x_invisible"
        # DGPO drop-in flags: this encoder needs BOTH the pooled CLS token and the
        # per-object token set forwarded in batch_kb.
        self.requires_event_token = True
        self.requires_object_token = True
        self.reconstructs_raw_neutrinos = True
        # res_mse/* = per-component physical-space neutrino reconstruction residual (MSE)
        # vs truth (pt,eta,phi spherical | px,py,pz cartesian) -- the "is the error going
        # down?" diagnostic. Aggregated over val, logged in its own residual/ W&B section.
        self._res_names = ("px", "py", "pz") if self.cartesian else ("pt", "eta", "phi")
        self.metric_keys = ["recon_nu_mse", "recon_token_mse"]
        for _nm in self._res_names:
            self.metric_keys.append(f"res_mse/{_nm}")

        dev = torch.device(device) if device is not None else None
        if normalizers is None:
            seq_norm, glob_norm, inv_norm = load_normalizers_from_pt(
                normalization_file, device=dev, cartesian=self.cartesian
            )
        else:
            seq_norm, glob_norm, inv_norm = normalizers
        self.sequential_normalizer = seq_norm
        self.global_normalizer = glob_norm
        self.invisible_normalizer = inv_norm

        n_inv = int(self.invisible_normalizer.mean.numel())
        if nu_kin_dim != n_inv:
            raise ValueError(f"nu_kin_dim={nu_kin_dim} != invisible normalizer dim {n_inv}")

        self.token_dim = int(token_dim)
        self.nu_kin_dim = int(nu_kin_dim)
        self.d_model = int(d_model) if d_model is not None else self.token_dim
        self.latent_dim = int(latent_dim)

        if self.cartesian:
            phi_index = None
        self.phi_index = None if phi_index is None else int(phi_index)
        self.invisible_normalizer.inv_cdf_index = (
            [] if self.phi_index is None else [self.phi_index]
        )

        self.hparams: dict[str, Any] = {
            "model_type": MODEL_TYPE,
            "token_dim": self.token_dim,
            "nu_kin_dim": self.nu_kin_dim,
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "ffn_mult": int(ffn_mult),
            "dropout": float(dropout),
            "cartesian": self.cartesian,
            "phi_index": self.phi_index,
        }

        # Standardization buffers: separate stats for the pooled CLS token and the
        # per-object tokens (same ObjectEncoder output space but different
        # distributions). Identity until set via set_token_stats / set_object_token_stats.
        self.register_buffer("token_mean", torch.zeros(self.token_dim))
        self.register_buffer("token_std", torch.ones(self.token_dim))
        self.register_buffer("obj_token_mean", torch.zeros(self.token_dim))
        self.register_buffer("obj_token_std", torch.ones(self.token_dim))

        hidden = max(self.d_model, 2 * self.nu_kin_dim)
        self.nu_embed = _mlp(self.nu_kin_dim, hidden, self.d_model, dropout)
        self.antinu_embed = _mlp(self.nu_kin_dim, hidden, self.d_model, dropout)
        # Shared linear embed for CLS *and* per-object tokens (same feature space);
        # type embeddings disambiguate roles.
        self.token_embed = nn.Linear(self.token_dim, self.d_model)

        # Types: 0=z-cls, 1=event-CLS, 2=object, 3=nu, 4=anti-nu.
        self.z_cls = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(5, self.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_mult) * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(num_layers), enable_nested_tensor=False
        )

        self.latent_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.latent_dim),
        )
        # Non-conditional decoder: z alone -> [nu pair (2*kin) ; event CLS token (D)].
        # Object tokens are NOT reconstructed (input-only context).
        out_dim = 2 * self.nu_kin_dim + self.token_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, out_dim),
        )

        if dev is not None:
            self.to(dev)

    # ------------------------------------------------------------------ token stats
    @torch.no_grad()
    def set_token_stats(self, mean: Tensor, std: Tensor) -> None:
        """Install per-dim event-CLS-token standardization (train-set statistics)."""
        if mean.numel() != self.token_dim or std.numel() != self.token_dim:
            raise ValueError(
                f"token stats must have dim {self.token_dim}, got {mean.numel()}/{std.numel()}"
            )
        self.token_mean.copy_(mean.reshape(-1).to(self.token_mean))
        self.token_std.copy_(std.reshape(-1).clamp_min(1e-6).to(self.token_std))

    @torch.no_grad()
    def set_object_token_stats(self, mean: Tensor, std: Tensor) -> None:
        """Install per-dim per-object-token standardization (over valid objects)."""
        if mean.numel() != self.token_dim or std.numel() != self.token_dim:
            raise ValueError(
                f"object token stats must have dim {self.token_dim}, "
                f"got {mean.numel()}/{std.numel()}"
            )
        self.obj_token_mean.copy_(mean.reshape(-1).to(self.obj_token_mean))
        self.obj_token_std.copy_(std.reshape(-1).clamp_min(1e-6).to(self.obj_token_std))

    def _standardize_token(self, token: Tensor) -> Tensor:
        return (token - self.token_mean) / self.token_std

    def _standardize_object(self, obj: Tensor) -> Tensor:
        return (obj - self.obj_token_mean) / self.obj_token_std

    # ------------------------------------------------------------------ inputs
    def event_token_from_batch(self, batch: Mapping[str, Tensor]) -> Tensor:
        tok = batch.get("event_token")
        if tok is None:
            raise KeyError(
                "batch must contain 'event_token' (augmented parquet from "
                "preprocessing/augment_event_token.py)"
            )
        if tok.dim() != 2 or tok.shape[-1] != self.token_dim:
            raise ValueError(
                f"event_token must be (B, {self.token_dim}), got {tuple(tok.shape)}"
            )
        return tok

    def object_token_from_batch(self, batch: Mapping[str, Tensor]) -> Tensor:
        obj = batch.get("object_token")
        if obj is None:
            raise KeyError(
                "batch must contain 'object_token' (augmented parquet from "
                "preprocessing/augment_event_token.py --object-tokens)"
            )
        # Accept both storage layouts: the P*D scalar-column form -> (B, P, D), and the
        # packed FixedSizeList form (preprocessing/pack_object_token.py) which arrives as
        # (B, P*D). float16 (float16 repack) is upcast. All handled here so the encoder is
        # storage-format agnostic.
        obj = obj.float()
        if obj.dim() == 2 and obj.shape[-1] % self.token_dim == 0:
            obj = obj.reshape(obj.shape[0], -1, self.token_dim)  # packed (B, P*D) -> (B, P, D)
        if obj.dim() != 3 or obj.shape[-1] != self.token_dim:
            raise ValueError(
                f"object_token must be (B, P, {self.token_dim}) or packed (B, P*{self.token_dim}), "
                f"got {tuple(obj.shape)}"
            )
        return obj

    def object_mask_from_batch(self, batch: Mapping[str, Tensor], num_obj: int) -> Tensor:
        """Bool ``(B, P)`` valid-object mask from ``x_mask`` (True = real object)."""
        m = batch.get("x_mask")
        if m is None:
            raise KeyError("batch must contain 'x_mask' for object-token padding")
        if m.dim() == 3 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        if m.shape[-1] != num_obj:
            raise ValueError(
                f"x_mask objects {m.shape[-1]} != object_token objects {num_obj}"
            )
        return m.bool()

    def neutrino_kin_from_batch(self, batch: Mapping[str, Tensor]) -> Tensor:
        if "nu_kin" in batch:
            kin = batch["nu_kin"]
        elif self.invisible_key in batch:
            t = batch[self.invisible_key]
            kin = t if t.shape[-1] == self.nu_kin_dim else t[..., _NU_KIN_SLICE_7]
        elif "x_invisible" in batch:
            t = batch["x_invisible"]
            kin = t if t.shape[-1] == self.nu_kin_dim else t[..., _NU_KIN_SLICE_7]
        else:
            raise KeyError(f"batch must contain 'nu_kin' or '{self.invisible_key}'")
        if kin.shape[-2] != 2 or kin.shape[-1] != self.nu_kin_dim:
            raise ValueError(
                f"neutrino kin must be (B, 2, {self.nu_kin_dim}), got {tuple(kin.shape)}"
            )
        return kin

    def _normalize_neutrinos(self, nu_kin: Tensor) -> Tensor:
        """Differentiable mirror of ``Normalizer.forward`` (matches DGPO/EveNet)."""
        inv = self.invisible_normalizer
        out = (nu_kin - inv.mean) / inv.std
        if len(inv.inv_cdf_index) > 0:
            idx = torch.as_tensor(inv.inv_cdf_index, device=nu_kin.device, dtype=torch.long)
            partial = out.index_select(-1, idx)
            partial = ((partial + math.sqrt(3)) / (2 * math.sqrt(3))).clamp(1e-6, 1 - 1e-6)
            partial = inv.normal.icdf(partial)
            out = out.index_copy(-1, idx, partial.to(out.dtype))
        return out

    def denormalize_neutrinos(self, nu_norm: Tensor) -> Tensor:
        inv = self.invisible_normalizer
        out = nu_norm
        if len(inv.inv_cdf_index) > 0:
            idx = torch.as_tensor(inv.inv_cdf_index, device=nu_norm.device, dtype=torch.long)
            partial = inv.normal.cdf(out.index_select(-1, idx))
            partial = partial * (2 * math.sqrt(3)) - math.sqrt(3)
            out = out.index_copy(-1, idx, partial.to(out.dtype))
        return out * inv.std + inv.mean

    # ------------------------------------------------------------------ encode / decode
    def encode_latent(
        self, batch: Mapping[str, Tensor], *, detach_neutrinos: bool = False
    ) -> Tensor:
        """Encode (event token, object tokens, nu pair) -> bottleneck ``z``.

        Not ``@torch.no_grad()``: with a frozen encoder, ``z_pred`` stays
        differentiable w.r.t. the DGPO-predicted neutrinos. The event/object tokens
        are frozen context and enter detached (no grad path through them).
        """
        token = self.event_token_from_batch(batch)              # (B, D)
        obj = self.object_token_from_batch(batch)               # (B, P, D)
        num_obj = int(obj.shape[1])
        obj_valid = self.object_mask_from_batch(batch, num_obj)  # (B, P) bool
        nu_kin = self.neutrino_kin_from_batch(batch)
        if detach_neutrinos:
            nu_kin = nu_kin.detach()
        bsz = nu_kin.shape[0]

        nu_norm = self._normalize_neutrinos(nu_kin)                       # (B, 2, kd)
        tok_std = self._standardize_token(token.detach())                # (B, D)
        obj_std = self._standardize_object(obj.detach())                 # (B, P, D)

        evt_tok = self.token_embed(tok_std).unsqueeze(1) + self.type_embed[1]   # (B,1,d)
        obj_tok = self.token_embed(obj_std) + self.type_embed[2]               # (B,P,d)
        nu_tok = self.nu_embed(nu_norm[:, 0, :]).unsqueeze(1) + self.type_embed[3]
        antinu_tok = self.antinu_embed(nu_norm[:, 1, :]).unsqueeze(1) + self.type_embed[4]
        zc = self.z_cls.expand(bsz, 1, self.d_model) + self.type_embed[0]

        seq = torch.cat([zc, evt_tok, obj_tok, nu_tok, antinu_tok], dim=1)  # (B, P+4, d)
        # key_padding_mask: True = ignore. z-cls, event-CLS, nu, anti-nu always valid;
        # only padded objects are masked out.
        always = torch.zeros(bsz, 2, dtype=torch.bool, device=seq.device)
        pad = ~obj_valid  # (B, P) True where object is padding
        key_padding_mask = torch.cat([always, pad, always], dim=1)  # (B, P+4)
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        return self.latent_head(out[:, 0, :])  # (B, latent_dim)

    def decode(self, z: Tensor) -> tuple[Tensor, Tensor]:
        out = self.decoder(z)
        nu_flat, tok = out[:, : 2 * self.nu_kin_dim], out[:, 2 * self.nu_kin_dim:]
        return nu_flat.reshape(z.shape[0], 2, self.nu_kin_dim), tok

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        z = self.encode_latent(batch)
        nu_reco, _ = self.decode(z)
        return z, nu_reco

    # ------------------------------------------------------------------ loss
    @torch.no_grad()
    def _neutrino_residuals(
        self, nu_reco_norm: Tensor, batch: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        """Per-component PHYSICAL MSE of reconstructed vs truth neutrinos.

        Denormalizes the reconstruction back to physical space and reports the squared
        error per kinematic component, masked to valid neutrino slots. Spherical:
        (pt = expm1(log_pt), eta, phi with 2*pi wrap). Cartesian: (px, py, pz).
        Diagnostic only -- no gradient.
        """
        reco = self.denormalize_neutrinos(nu_reco_norm.detach())      # (B, 2, kd) physical
        truth = self.neutrino_kin_from_batch(batch).detach()          # (B, 2, kd) physical
        if self.cartesian:
            diff = reco - truth
        else:
            d_pt = torch.expm1(reco[..., 0].clamp(-10.0, 10.0)) - torch.expm1(
                truth[..., 0].clamp(-10.0, 10.0)
            )
            d_eta = reco[..., 1] - truth[..., 1]
            d_phi = reco[..., 2] - truth[..., 2]
            d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))   # wrap to [-pi, pi]
            diff = torch.stack([d_pt, d_eta, d_phi], dim=-1)          # (B, 2, kd)

        slot_mask = batch.get("x_invisible_mask")
        if slot_mask is not None:
            m = slot_mask.to(diff.dtype).unsqueeze(-1)                # (B, 2, 1)
            denom = m.sum().clamp_min(1.0)
            mse = ((diff ** 2) * m).sum(dim=(0, 1)) / denom           # (kd,)
        else:
            mse = (diff ** 2).mean(dim=(0, 1))

        return {f"res_mse/{nm}": mse[i] for i, nm in enumerate(self._res_names)}

    def reconstruction_loss(
        self, batch: Mapping[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """MSE(nu) + MSE(event CLS token) -- plain sum, no weight (both unit-variance)."""
        z = self.encode_latent(batch)
        nu_reco, tok_reco = self.decode(z)

        nu_target = self._normalize_neutrinos(self.neutrino_kin_from_batch(batch)).detach()
        tok_target = self._standardize_token(self.event_token_from_batch(batch)).detach()

        sq_nu = (nu_reco - nu_target) ** 2
        slot_mask = batch.get("x_invisible_mask")
        if slot_mask is not None:
            m = slot_mask.float().unsqueeze(-1)  # (B, 2, 1)
            denom = (m.sum() * self.nu_kin_dim).clamp_min(1.0)
            nu_loss = (sq_nu * m).sum() / denom
        else:
            nu_loss = sq_nu.mean()
        tok_loss = ((tok_reco - tok_target) ** 2).mean()

        # Both targets are unit-variance standardized and mean-reduced, so the two MSEs
        # share one dimensionless unit -> plain sum (implicit weight 1, no hyperparameter).
        loss = nu_loss + tok_loss
        metrics = {
            "recon_mse": loss.detach(),
            "recon_nu_mse": nu_loss.detach(),
            "recon_token_mse": tok_loss.detach(),
            "latent_rms": z.detach().pow(2).mean().sqrt(),
        }
        # Supervisor-requested diagnostic: per-component physical reconstruction
        # residual (MSE) vs truth -- confirms the error is actually shrinking.
        metrics.update(self._neutrino_residuals(nu_reco, batch))
        return loss, metrics


# ----------------------------------------------------------------------
# checkpoint I/O
# ----------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: ObjectTokenBottleneckAutoencoder,
    *,
    normalization_file: str | Path,
    epoch: int | None = None,
    val_loss: float | None = None,
    global_step: int | None = None,
    lc_next_epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save a self-describing training checkpoint (rank-0 caller's responsibility).

    Payload mirrors DGPO / EveNet scheduling keys where useful:

    - ``global_step`` — optimizer steps completed (W&B x-axis, resume counter).
    - ``lc_next_epoch`` — next epoch index the loop should run (like ``dgpo_next_epoch``).
    - ``optimizer_state_dict`` / ``scheduler_state_dict`` — full resume.

    Token standardization buffers (``token_mean/std``, ``obj_token_mean/std``)
    persist inside ``model_state_dict``, so the frozen DGPO encoder reuses the
    exact training-time statistics.
    """
    payload: dict[str, Any] = {
        "latent_constraint_version": _CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "hparams": dict(model.hparams),
        "normalization_file": str(normalization_file),
        "epoch": epoch,
        "val_loss": val_loss,
        "global_step": int(global_step) if global_step is not None else None,
        "lc_next_epoch": int(lc_next_epoch) if lc_next_epoch is not None else None,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(dict(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def parse_lc_resume_from_checkpoint(checkpoint: dict[str, Any] | None) -> tuple[int, int]:
    """Return ``(start_epoch, global_step)`` for the training loop.

    Mirrors :func:`RL.DGPO_neutrino.model_utils.parse_dgpo_resume_from_checkpoint`:
    weights are loaded separately; this only interprets scheduling counters.
    """
    if not checkpoint:
        return 0, 0
    gs = int(checkpoint.get("global_step", 0))
    if int(checkpoint.get("latent_constraint_version", 0)) >= 1:
        if "lc_next_epoch" in checkpoint and checkpoint["lc_next_epoch"] is not None:
            return int(checkpoint["lc_next_epoch"]), gs
        return 0, gs
    ep = int(checkpoint.get("epoch", -1))
    return (ep + 1) if ep >= 0 else 0, gs


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    normalization_file: str | Path | None = None,
) -> tuple[ObjectTokenBottleneckAutoencoder, dict[str, Any]]:
    """Load a checkpoint into a freshly built :class:`ObjectTokenBottleneckAutoencoder`.

    Args:
        path: checkpoint path written by :func:`save_checkpoint`.
        device: target device.
        normalization_file: override for the stored normalization path (useful
            when the training-time absolute path is unavailable).

    Returns:
        ``(model, cfg)`` where ``cfg`` is the stored payload (minus weights). The
        model is on ``device`` but **not** set to ``eval()`` / frozen — the DGPO
        caller does that explicitly.
    """
    dev = torch.device(device)
    ckpt: dict[str, Any] = torch.load(str(path), map_location=dev, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"{path} is not a latent-constraint checkpoint (missing model_state_dict). "
            "Use best.ckpt or last.ckpt from train_latent_constraint.py."
        )
    if "hparams" not in ckpt:
        raise KeyError(
            f"{path} is not a latent-constraint checkpoint (missing hparams). "
            "Use best.ckpt or last.ckpt from train_latent_constraint.py."
        )
    version = int(ckpt.get("latent_constraint_version", 0))
    if version < 1:
        raise ValueError(
            f"{path} has latent_constraint_version={version}; expected >= 1 "
            "(re-train or export with current train_latent_constraint.py)."
        )
    hparams = dict(ckpt["hparams"])
    norm_file = normalization_file or ckpt.get("normalization_file")
    if not norm_file:
        raise ValueError("checkpoint has no normalization_file; pass one explicitly")

    model_type = str(hparams.get("model_type", ""))
    if model_type != MODEL_TYPE:
        raise ValueError(
            f"{path} has model_type={model_type!r}; only {MODEL_TYPE!r} is supported "
            "(older latent-constraint variants were removed — re-train with "
            "train_latent_constraint.py)."
        )
    model = ObjectTokenBottleneckAutoencoder(
        normalization_file=norm_file,
        token_dim=hparams["token_dim"],
        nu_kin_dim=hparams["nu_kin_dim"],
        d_model=hparams["d_model"],
        latent_dim=hparams["latent_dim"],
        num_layers=hparams["num_layers"],
        num_heads=hparams["num_heads"],
        ffn_mult=hparams["ffn_mult"],
        dropout=hparams["dropout"],
        cartesian=hparams.get("cartesian", False),
        phi_index=hparams.get("phi_index", 2),
        device=dev,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(dev)
    cfg = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    return model, cfg
