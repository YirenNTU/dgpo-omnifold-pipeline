"""Diffusion-only query-dependent readout of unpooled observed tokens."""
import torch
from torch import nn


class TokenSpecificConditioning(nn.Module):
    def __init__(self, hidden_dim, memory_dim, *, width=64, heads=4):
        super().__init__()
        if width < 2 or heads < 1 or width % heads:
            raise ValueError("token conditioning width must be divisible by positive heads")
        self.query = nn.Sequential(nn.LayerNorm(hidden_dim, elementwise_affine=False),
                                   nn.Linear(hidden_dim, width))
        self.memory = nn.Linear(memory_dim, width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.encoder = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(),
                                     nn.LayerNorm(width, elementwise_affine=False))
        # One zero output gate; attention/MLP must remain normally initialized.
        self.output = nn.Linear(width, 4 * hidden_dim)
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
        with torch.no_grad():
            parts = packed.detach().float().reshape(*packed.shape[:2], 2, 2, -1)
            count = (valid.sum() * 2 * parts.shape[-1]).clamp_min(1)
            self.diagnostics = {
                "token_scale_rms": (parts[..., 0, :].square().sum() / count).sqrt(),
                "token_shift_rms": (parts[..., 1, :].square().sum() / count).sqrt(),
            }
        return packed.chunk(4, dim=-1)
