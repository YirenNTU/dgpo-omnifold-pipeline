"""Conditional heads; optional attention adds capacity but no new observables."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scripts.train_conditional_spin_ratio import ConditionalSpinMLP


class ConditioningBlock(nn.Module):
    def __init__(self, kind, width=64, dropout=.05, context_width=None):
        super().__init__()
        if kind not in ('concat', 'film'):
            raise ValueError('Expected concat or film')
        self.kind = kind
        context_width = width if context_width is None else context_width
        self.norm = nn.LayerNorm(width)
        self.context = nn.Linear(width, 2*width, bias=False)
        self.hidden = nn.Linear(width, 2*width)
        self.output = nn.Linear(2*width, width)
        self.dropout = nn.Dropout(dropout)
        # Equal initial functions; no unused padding parameters or zero output
        # gate. The context projection receives gradients immediately.
        nn.init.zeros_(self.context.weight)
        if context_width != width:
            # Preserve the RNG stream and all shared hidden parameters.
            with torch.random.fork_rng(devices=[]):
                self.context = nn.Linear(context_width, 2*width, bias=False)
                nn.init.zeros_(self.context.weight)

    def forward(self, h, c):
        z, context = self.norm(h), self.context(c)
        if self.kind == 'concat':
            # Exactly Linear(cat([z,c])) with cat([W_hidden,W_context]).
            z = self.hidden(z) + context
        else:
            gamma, beta = context.chunk(2, dim=-1)
            z = self.hidden((1 + gamma)*z + beta)
        return self.output(self.dropout(F.silu(z)))


class ConditionedTauMLP(ConditionalSpinMLP):
    def __init__(self, condition_dim, hidden=128, dropout=.05, candidate_dim=15,
                 relative_dim=0, kind='concat', depth=3, ratio_bound=None,
                 condition_hidden=None, condition_width=64):
        if depth < 1:
            raise ValueError('Head depth must be positive')
        super().__init__(condition_dim, hidden, dropout, candidate_dim, relative_dim)
        condition_hidden = hidden if condition_hidden is None else condition_hidden
        if condition_hidden < 1 or condition_width < 1:
            raise ValueError('Condition dimensions must be positive')
        # Preserve the encoders and relative-input initialization in BOTH arms.
        del self.head
        self.blocks = nn.ModuleList([ConditioningBlock(kind, 64, dropout, condition_width) for _ in range(depth)])
        self.readout = nn.Linear(64, 1)
        if condition_hidden != hidden or condition_width != 64:
            with torch.random.fork_rng(devices=[]):
                self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, condition_hidden), nn.SiLU(),
                    nn.Linear(condition_hidden, condition_width), nn.SiLU())
        self.residual_scale = 1/math.sqrt(depth)
        if ratio_bound is not None and (not math.isfinite(ratio_bound) or ratio_bound <= 1):
            raise ValueError('Invalid ratio bound')
        self.ratio_bound = ratio_bound

    def forward(self, condition, candidate):
        c, h = self.condition_encoder(condition), self.spin_encoder(candidate)
        h = self.condition_candidate(h, condition)
        for block in self.blocks:
            h = h + self.residual_scale*block(h, c)
        latent = self.readout(h).flatten()
        if self.ratio_bound is not None:
            from scripts.tau_bounded_ratio import bounded_log_ratio
            return bounded_log_ratio(latent,self.ratio_bound)
        return latent

    def condition_candidate(self, h, condition):
        return h


class AttentionConditionedTauMLP(ConditionedTauMLP):
    """Candidate query reads existing normalized visible x tokens, not PET tokens.

    Global condition encoder, candidate encoder and three FiLM blocks are retained.
    The residual starts at zero; no new observable or target enters the input.
    """
    def __init__(self, *args, packing_spec, attention_heads=4, **kwargs):
        super().__init__(*args, **kwargs)
        from scripts.conditional_tau_preprocessing import layout
        spans, packed_width = layout(packing_spec)
        self.x_span, shape = spans['x']
        self.mask_span, mask_shape = spans['x_mask']
        if len(shape) != 2 or mask_shape != (shape[0],):
            raise ValueError('Attention requires particle x and aligned x_mask')
        if self.condition_encoder[0].in_features != packed_width + 16:
            raise ValueError('Attention condition packing mismatch')
        if attention_heads != 4 or len(self.blocks) != 3:
            raise ValueError('This ablation fixes three FiLM blocks and four attention heads')
        self.token_shape = shape
        with torch.random.fork_rng(devices=[]):
            # Extra module initialization must not shift shared dropout RNG.
            self.token_encoder = nn.Sequential(nn.Linear(shape[1], 64), nn.SiLU(), nn.LayerNorm(64))
            self.slot_identity = nn.Embedding(shape[0], 64)
            self.query_norm = nn.LayerNorm(64)
            self.cross_attention = nn.MultiheadAttention(64, attention_heads, dropout=0., batch_first=True)
            self.attention_output = nn.Linear(64, 64)
            nn.init.zeros_(self.attention_output.weight)
            nn.init.zeros_(self.attention_output.bias)

    def condition_candidate(self, h, condition):
        valid = condition[:, self.mask_span] > .5
        x = condition[:, self.x_span].reshape(len(condition), *self.token_shape)
        x = x.masked_fill(~valid[..., None], 0)
        tokens = self.token_encoder(x) + self.slot_identity.weight[None]
        # A dummy unmasked zero token makes empty events finite. Its output is
        # suppressed, so empty events receive exactly zero attention residual.
        nonempty = valid.any(dim=1)
        safe = valid.clone()
        safe[:, 0] |= ~nonempty
        tokens = tokens.masked_fill(~valid[..., None], 0)
        update, _ = self.cross_attention(self.query_norm(h)[:, None], tokens, tokens,
                                          key_padding_mask=~safe, need_weights=False)
        return h + self.attention_output(update[:, 0]) * nonempty[:, None]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
