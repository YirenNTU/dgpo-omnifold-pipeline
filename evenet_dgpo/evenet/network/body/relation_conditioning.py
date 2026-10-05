"""Visible pair summaries for an opt-in, identity-initialized FiLM adapter."""
import torch
from torch import nn


def pair_summary(theta, phi, energy, pt, valid):
    """Symmetric unordered-pair mean/max; O(B*N) temporary memory.

    Features: cos(delta phi), opening cosine, squared energy and pT asymmetry.
    Inputs are observed particles only; no truth, candidate, or tau assignment.
    """
    clean = lambda x: torch.where(valid, x, torch.zeros_like(x))
    theta, phi, energy, pt = map(clean, (theta, phi, energy, pt))
    if any(not torch.isfinite(x).all() for x in (theta, phi, energy, pt)):
        raise ValueError('Nonfinite valid visible relation input')
    energy, pt = energy.clamp_min(0), pt.clamp_min(0)
    total = theta.new_zeros(theta.shape[0], 4)
    maximum = torch.full_like(total, -torch.inf)
    count = theta.new_zeros(theta.shape[0], 1)
    for i in range(theta.shape[1]-1):
        keep = valid[:,i:i+1] & valid[:,i+1:]
        cosphi = (phi[:,i:i+1]-phi[:,i+1:]).cos()
        opening = theta[:,i:i+1].cos()*theta[:,i+1:].cos()+theta[:,i:i+1].sin()*theta[:,i+1:].sin()*cosphi
        asym = lambda x: ((x[:,i:i+1]-x[:,i+1:])/(x[:,i:i+1]+x[:,i+1:]).clamp_min(1e-8)).square()
        features = torch.stack((cosphi, opening.clamp(-1,1), asym(energy), asym(pt)), -1)
        total = total + torch.where(keep[...,None],features,0.).sum(1)
        maximum = torch.maximum(maximum, features.masked_fill(~keep[...,None],-torch.inf).amax(1))
        count = count + keep.sum(1,keepdim=True)
    present = count > 0
    maximum = torch.where(present, maximum, 0.)
    return torch.cat((total/count.clamp_min(1), maximum),-1), present


class RelationConditioning(nn.Module):
    def __init__(self, context_dim, output_dim, layers, mode='relations', width=64, seed=20260930):
        super().__init__()
        if mode not in ('relations','context') or context_dim < 8 or width < 8:
            raise ValueError('Invalid relation adapter configuration')
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(context_dim+8,width),nn.SiLU(),
                nn.Linear(width,width),nn.SiLU(),nn.LayerNorm(width,elementwise_affine=False))
            self.outputs = nn.ModuleList(nn.Linear(width,output_dim) for _ in range(layers))
            for head in self.outputs:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, context, relations, present):
        # Same active parameter count and graph; context control sees no pair geometry.
        extra = relations if self.mode == 'relations' else context[:,:8]
        encoded = self.encoder(torch.cat((context,extra.to(context)),-1))
        return [head(encoded)*present.to(context) for head in self.outputs]


def load_relation_weights(model, checkpoint):
    """Load all shared tensors exactly, permit only configured zero-output additions."""
    prefix = 'TruthGeneration.visible_conditioning.relation_adapter.'
    pair_prefix = 'TruthGeneration.visible_conditioning.pair_attention.'
    conditioning = model.TruthGeneration.visible_conditioning
    prefixes = (prefix, pair_prefix) if conditioning.pair_attention is not None else (prefix,)
    preconditioner = getattr(model, 'conditional_preconditioning', None)
    if preconditioner is not None:
        if not preconditioner.identity_initialized():
            raise ValueError('Load the raw source before fitted conditional preconditioning')
        prefixes += ('conditional_preconditioning.',)
    source = {k.removeprefix('model.'):v for k,v in checkpoint['state_dict'].items()}
    target = model.state_dict()
    if any(k.startswith((prefix, pair_prefix, 'conditional_preconditioning.')) for k in source):
        raise ValueError('Use a source before relation-adapter / pair-attention training')
    added = [k for k in target if k not in source]
    missing_shared = [k for k in added if not k.startswith(prefixes)]
    # Supervised checkpoints include FAMO loss-balancing weights, installed by
    # the engine after this load. Match the successful raw diagnostic loader:
    # ignore ONLY those auxiliary entries absent from the current model.
    unexpected = sorted(k for k in source if k not in target and not k.startswith('famo.w.'))
    if not added or missing_shared or unexpected:
        raise ValueError('Incompatible relation source: '
                         f'new_adapter_tensors={sum(k.startswith(prefix) for k in added)}, '
                         f'missing_shared={missing_shared}, unexpected={unexpected}')
    source = {k: v for k, v in source.items() if k in target}
    for k,v in source.items():
        if v.shape != target[k].shape or v.dtype != target[k].dtype or not torch.isfinite(v).all():
            raise ValueError(f'Incompatible shared tensor: {k}')
    adapter = model.TruthGeneration.visible_conditioning.relation_adapter
    if any(torch.count_nonzero(p) for h in adapter.outputs for p in h.parameters()):
        raise ValueError('New modulation outputs must be zero')
    if conditioning.pair_attention is not None:
        if any(torch.count_nonzero(p) for h in conditioning.pair_attention.outputs for p in h.parameters()):
            raise ValueError('New attention biases must be zero')
    model.load_state_dict({**target,**source},strict=True)
    if preconditioner is not None:
        # The source checkpoint, not a possibly updated normalization file,
        # defines the chart used to fit and later apply the coordinate model.
        for name in ('sequential_normalizer', 'global_normalizer'):
            getattr(preconditioner, name).load_state_dict(getattr(model, name).state_dict(), strict=True)
        preconditioner.target_normalizer.load_state_dict(model.invisible_normalizer.state_dict(), strict=True)
