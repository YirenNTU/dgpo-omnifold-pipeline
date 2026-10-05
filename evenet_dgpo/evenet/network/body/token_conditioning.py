"""Diffusion-only query-dependent readout of unpooled observed tokens."""
import torch
from torch import nn


class TokenSpecificConditioning(nn.Module):
    def __init__(self, hidden_dim, memory_dim, *, width=64, heads=4,
                 mode="film", block="last", output_dim=None):
        super().__init__()
        if width < 2 or heads < 1 or width % heads:
            raise ValueError("token conditioning width must be divisible by positive heads")
        if mode not in ("film", "residual", "velocity") or block not in ("last", "all", "output"):
            raise ValueError("token conditioning requires a supported mode and injection site")
        if mode == "film" and block != "last":
            raise ValueError("legacy token FiLM supports only the last diffusion block")
        if mode == "residual" and block not in ("last", "all"):
            raise ValueError("hidden residual conditioning supports last/all blocks")
        if mode == "velocity" and (block != "output" or not output_dim or int(output_dim) < 1):
            raise ValueError("velocity conditioning requires block=output and a positive output_dim")
        self.mode, self.block = mode, block
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim, elementwise_affine=False),
                                   nn.Linear(hidden_dim, width))
        self.memory = nn.Linear(memory_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.encoder = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(),
                                     nn.LayerNorm(width, elementwise_affine=False))
        # One zero output gate; attention/MLP must remain normally initialized.
        output_width = (4 * hidden_dim if mode == "film" else
                        int(output_dim) if mode == "velocity" else hidden_dim)
        self.output = nn.Linear(width, output_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.diagnostics = {}

    def forward(self, queries, memory, query_mask, memory_mask):
        qvalid = query_mask.squeeze(-1).bool() if query_mask.ndim == 3 else query_mask.bool()
        mvalid = memory_mask.squeeze(-1).bool() if memory_mask.ndim == 3 else memory_mask.bool()
        if qvalid.shape != queries.shape[:2] or mvalid.shape != memory.shape[:2] or not memory.shape[1]:
            raise ValueError("token conditioning requires matching query/memory masks and padded memory")
        q = self.query(torch.where(qvalid[..., None], queries, 0.))
        kv = self.memory(torch.where(mvalid[..., None], memory, 0.))
        present = mvalid.any(dim=1)
        safe_mask = mvalid.clone()
        safe_mask[:, 0] |= ~present  # finite sentinel for events with no visible particles
        readout, _ = self.attention(q, kv, kv, key_padding_mask=~safe_mask, need_weights=False)
        packed = self.output(self.encoder(torch.cat((q, readout), dim=-1)))
        valid = qvalid & present[:, None]
        packed = torch.where(valid[..., None], packed, 0.)
        if self.mode in ("residual", "velocity"):
            with torch.no_grad():
                count = (valid.sum() * packed.shape[-1]).clamp_min(1)
                residual_rms = (packed.detach().float().square().sum() / count).sqrt()
                hidden = torch.where(valid[..., None], queries.detach().float(), 0.)
                hidden_rms = (hidden.square().sum() / count).sqrt()
                if self.mode == "residual":
                    self.diagnostics = {
                        "token_residual_rms": residual_rms,
                        "token_residual_to_hidden_rms": residual_rms / hidden_rms.clamp_min(1e-8),
                    }
                else:
                    self.diagnostics = {"velocity_residual_rms": residual_rms}
            return packed
        with torch.no_grad():
            parts = packed.detach().float().reshape(*packed.shape[:2], 2, 2, -1)
            count = (valid.sum() * 2 * parts.shape[-1]).clamp_min(1)
            self.diagnostics = {
                "token_scale_rms": (parts[..., 0, :].square().sum() / count).sqrt(),
                "token_shift_rms": (parts[..., 1, :].square().sum() / count).sqrt(),
            }
        return packed.chunk(4, dim=-1)
