"""Shared per-leg observed-momentum adapter, added only before attention.

No truth, candidate, process-ID shortcut or reconstructed tau enters this branch.
Both legs use the same MLP and the same momentum normalization, derived from
the original pinned normalizer. The original policy normalizer is unchanged.
"""
import torch
from torch import nn
from .met_conditioning import FIELDS

PREFIX = 'TruthGeneration.visible_conditioning.leg_attention_conditioning.'


def leg_spec(options, global_features):
    options = dict(options or {})
    if not options.get('enabled', False):
        return None
    if set(options) - {'enabled', 'width', 'seed', 'attention_route'} or options['enabled'] is not True:
        raise ValueError('Unknown leg attention conditioning option')
    names = list(global_features or [])
    if len(names) != len(set(names)) or not set(FIELDS).issubset(names):
        raise ValueError('Leg attention requires observed missing and lead_a/b momenta')
    width, seed = options.get('width', 128), options.get('seed', 42)
    if type(width) is not int or width < 2 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid leg attention width/seed')
    route = options.get('attention_route', 'qkv')
    if route not in ('qkv', 'qk'):
        raise ValueError('Leg attention route must be qkv or qk')
    result = dict(version=1, fields=list(FIELDS), global_features=names,
                indices=[names.index(k) for k in FIELDS], width=width, seed=seed,
                normalization='pinned_missing_and_shared_leading_momentum',
                inputs='event_context,own_visible_xyz,other_visible_xyz,missing_xyz',
                target='attention', shared_leg_mlp=True)
    if route == 'qk':
        result.update(version=2, attention_route='qk')
    return result


class LegAttentionConditioning(nn.Module):
    def __init__(self, context_dim, hidden_dim, layers, spec):
        super().__init__()
        if spec != leg_spec(dict(enabled=True, width=spec['width'], seed=spec['seed'],
                                 attention_route=spec.get('attention_route', 'qkv')), spec['global_features']):
            raise ValueError('Leg attention feature schema changed')
        self.spec = dict(spec)
        if spec.get('attention_route') == 'qk':
            # Persist semantics: otherwise identical tensor shapes could load a
            # trained Q/K/V adapter with the wrong routing without an error.
            self.register_buffer('attention_route', torch.tensor(1, dtype=torch.int8))
        self.register_buffer('indices', torch.tensor(spec['indices'], dtype=torch.long))
        self.register_buffer('source_mean', torch.zeros(9))
        self.register_buffer('source_std', torch.ones(9))
        self.register_buffer('shared_mean', torch.zeros(3))
        self.register_buffer('shared_std', torch.ones(3))
        self.register_buffer('normalization_bound', torch.tensor(False))
        self._normalizer_bound = False
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed'])
            width = spec['width']
            self.encoder = nn.Sequential(nn.Linear(context_dim + 9, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU())
            self.modulations = nn.ModuleList(nn.Linear(width, 2 * hidden_dim) for _ in range(layers))
            for head in self.modulations:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    @torch.no_grad()
    def bind_normalizer(self, normalizer):
        if set(self.spec['indices']) & set(normalizer.inv_cdf_index):
            raise ValueError('Leg momentum normalization must be affine')
        mean = normalizer.mean.flatten().index_select(0, self.indices.to(normalizer.mean.device))
        std = normalizer.std.flatten().index_select(0, self.indices.to(normalizer.std.device))
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid pinned momentum normalization')
        self.source_mean.copy_(mean)
        self.source_std.copy_(std)
        common = (mean[3:6] + mean[6:9]) / 2
        scale = ((std[3:6].square() + std[6:9].square()) / 2
                 + (mean[3:6] - mean[6:9]).square() / 4).sqrt()
        self.shared_mean.copy_(common)
        self.shared_std.copy_(scale)
        self.normalization_bound.fill_(True)
        self._normalizer_bound = True

    def forward(self, context, normalized_globals, global_mask):
        if not self._normalizer_bound:
            raise ValueError('Bind the pinned policy normalizer before leg conditioning')
        n = len(context)
        if normalized_globals is None or global_mask is None:
            raise ValueError('Leg attention requires normalized observed globals and mask')
        if (normalized_globals.shape != (n, 1, len(self.spec['global_features']))
                or global_mask.shape != (n, 1, 1)):
            raise ValueError('Leg attention global shape/schema mismatch')
        valid = global_mask[:, 0].bool()
        selected = normalized_globals[:, 0].index_select(-1, self.indices)
        selected = torch.where(valid, selected, 0.).to(context)
        raw = selected * self.source_std.to(context) + self.source_mean.to(context)
        a = (raw[:, 3:6] - self.shared_mean.to(context)) / self.shared_std.to(context)
        b = (raw[:, 6:9] - self.shared_mean.to(context)) / self.shared_std.to(context)
        missing = selected[:, :3]
        leg_inputs = torch.stack((torch.cat((context, a, b, missing), -1),
                                  torch.cat((context, b, a, missing), -1)), 1)
        hidden = self.encoder(leg_inputs)
        return [torch.where(valid[:, None], head(hidden), 0.).chunk(2, -1)
                for head in self.modulations]


def load_leg_weights(model, checkpoint, *, allow_new=False):
    original = checkpoint['state_dict']
    source = {k.removeprefix('model.'): v for k, v in original.items()}
    if len(source) != len(original):
        raise ValueError('Checkpoint prefix collision')
    target = model.state_dict()
    added = {k for k in target if k.startswith(PREFIX)}
    saved = {k for k in source if k.startswith(PREFIX)}
    missing = set(target) - set(source)
    extra = {k for k in source if k not in target and not k.startswith('famo.w.')}
    if allow_new:
        if not added or saved or missing != added:
            raise ValueError('Leg upgrade permits exactly one zero branch on an intact source')
        branch = model.TruthGeneration.visible_conditioning.leg_attention_conditioning
        if not branch.normalization_bound or any(torch.count_nonzero(p) for h in branch.modulations for p in h.parameters()):
            raise ValueError('Leg extension requires pinned normalization and zero output')
    elif missing:
        raise ValueError('Missing leg/model weights; explicitly prepare an upgraded source')
    if extra:
        raise ValueError('Unexpected model weights: ' + str(sorted(extra)))
    buffer_keys = {PREFIX + name for name, _ in model.TruthGeneration.visible_conditioning.leg_attention_conditioning.named_buffers()}
    for k, v in target.items():
        if k not in source:
            continue
        if v.shape != source[k].shape:
            raise ValueError('Checkpoint shape mismatch: ' + k)
        if ('normalizer.' in k or k in buffer_keys) and not torch.equal(v.cpu(), source[k].cpu()):
            raise ValueError('Pinned normalization or leg schema mismatch: ' + k)
    model.load_state_dict({k: source[k] if k in source else v for k, v in target.items()}, strict=True)
    return dict(loaded_existing_tensors=len(target)-len(missing), added_tensors=sorted(missing),
                ignored_training_state=sorted(set(source)-set(target)))
