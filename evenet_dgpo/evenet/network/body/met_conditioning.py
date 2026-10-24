"""Observed missing momentum and leading-visible context for diffusion FiLM.

Inputs use the policy's pinned global normalizer. No generated candidate,
truth neutrino, tau energy assumption, or truth-derived feature enters here.
"""
import torch
from torch import nn

FIELDS = ('missing_px', 'missing_py', 'missing_pz',
          'lead_a_px', 'lead_a_py', 'lead_a_pz', 'lead_b_px', 'lead_b_py', 'lead_b_pz')
PREFIX = 'TruthGeneration.visible_conditioning.met_conditioning.'


def met_spec(options, global_features):
    options = dict(options or {})
    if not options.get('enabled', False):
        return None
    if set(options) - {'enabled', 'width', 'seed', 'target'} or options['enabled'] is not True:
        raise ValueError('Unknown MET conditioning option')
    names = list(global_features or [])
    if len(names) != len(set(names)) or not set(FIELDS).issubset(names):
        raise ValueError('MET FiLM requires missing and lead_a/b observed momentum fields')
    width, seed = options.get('width', 64), options.get('seed', 42)
    if type(width) is not int or width < 2 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid MET conditioning width/seed')
    target = options.get('target', 'both')
    if target not in ('both', 'attention'):
        raise ValueError('MET target must be both or attention')
    result = dict(version=1, fields=list(FIELDS), global_features=names,
                indices=[names.index(k) for k in FIELDS], width=width, seed=seed,
                normalization='pinned_policy_global_normalizer')
    if target != 'both': result['target'] = target
    return result


class METConditioning(nn.Module):
    def __init__(self, context_dim, output_dim, layers, spec):
        super().__init__()
        expected = met_spec(dict(enabled=True, width=spec['width'], seed=spec['seed'], target=spec.get('target', 'both')), spec['global_features'])
        if spec != expected:
            raise ValueError('MET feature schema changed')
        self.spec = dict(spec)
        self.attention_only = spec.get('target') == 'attention'
        if self.attention_only and output_dim % 4:
            raise ValueError('Attention MET requires attention/FFN gamma-beta layout')
        learned_dim = output_dim // 2 if self.attention_only else output_dim
        self.register_buffer('indices', torch.tensor(spec['indices'], dtype=torch.long))
        # Preserve all existing initialization and rollout random streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(spec['seed'])
            width = spec['width']
            self.encoder = nn.Sequential(nn.Linear(context_dim+len(FIELDS), width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU())
            self.modulations = nn.ModuleList(nn.Linear(width, learned_dim) for _ in range(layers))
            for head in self.modulations:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, context, normalized_globals, global_mask):
        n = len(context)
        if normalized_globals is None or global_mask is None:
            raise ValueError('MET FiLM requires normalized observed globals and their mask')
        if (normalized_globals.shape != (n, 1, len(self.spec['global_features']))
                or global_mask.shape != (n, 1, 1)):
            raise ValueError('MET global input shape/schema mismatch')
        valid = global_mask[:, 0].bool()
        selected = normalized_globals[:, 0].index_select(-1, self.indices)
        selected = torch.where(valid, selected, 0.).to(context)
        hidden = self.encoder(torch.cat((context, selected), -1))
        outputs = [torch.where(valid, head(hidden), 0.) for head in self.modulations]
        if self.attention_only:
            outputs = [torch.cat((x, torch.zeros_like(x)), -1) for x in outputs]
        return outputs


def load_met_weights(model, checkpoint, *, allow_new=False):
    """Exact load, permitting only an explicitly requested zero MET extension."""
    original = checkpoint['state_dict']
    source = {k.removeprefix('model.'):v for k,v in original.items()}
    if len(source) != len(original):raise ValueError('Checkpoint prefix collision')
    target = model.state_dict()
    added = {k for k in target if k.startswith(PREFIX)}
    saved = {k for k in source if k.startswith(PREFIX)}
    missing = set(target)-set(source)
    extra = {k for k in source if k not in target and not k.startswith('famo.w.')}
    if allow_new:
        if not added or saved or missing != added:
            raise ValueError('MET upgrade permits exactly one new zero branch on an intact source')
        branch = model.TruthGeneration.visible_conditioning.met_conditioning
        if any(torch.count_nonzero(p) for h in branch.modulations for p in h.parameters()):
            raise ValueError('MET extension must have zero output at initialization')
    elif missing:
        raise ValueError('Missing MET/model weights; explicitly prepare an upgraded source checkpoint')
    if extra:raise ValueError('Unexpected model weights: '+str(sorted(extra)))
    for k,v in target.items():
        if k not in source:continue
        if v.shape != source[k].shape:raise ValueError('Checkpoint shape mismatch: '+k)
        if ('normalizer.' in k or k == PREFIX+'indices') and not torch.equal(v.cpu(), source[k].cpu()):
            raise ValueError('Pinned normalization or MET schema mismatch: '+k)
    model.load_state_dict({k:source[k] if k in source else v for k,v in target.items()}, strict=True)
    return dict(loaded_existing_tensors=len(target)-len(missing), added_tensors=sorted(missing),
                ignored_training_state=sorted(set(source)-set(target)))


def assert_zero_met_extension(model, original):
    """An old frozen reward trunk may accompany an exactly equivalent new actor."""
    source, target = original.state_dict(), model.state_dict()
    if {k for k in target if not k.startswith(PREFIX)} != set(source):
        raise ValueError('Actor and original trunk differ beyond the MET extension')
    if any(not torch.equal(v.cpu(), target[k].cpu()) for k,v in source.items()):
        raise ValueError('Existing actor weights differ from the original reward denominator')
    branch = model.TruthGeneration.visible_conditioning.met_conditioning
    if branch is None or any(torch.count_nonzero(p) for h in branch.modulations for p in h.parameters()):
        raise ValueError('Initial MET branch is not function preserving')
