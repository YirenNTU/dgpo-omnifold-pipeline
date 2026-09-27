"""Local nonperiodic DGPO pilot. Fixed binned ratio, matched Fourier intervention."""
import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import conditioning_placement as placement
from . import conditioning_mse as core
from . import conditional as native


class Policy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_features(self, x, t, c):
        shape = x.shape[:-1]
        c = c.expand(*shape, 1).reshape(-1, 1)
        t = torch.as_tensor(t).expand(shape).reshape(-1)
        _, h, _, _ = self.model.representations(x.reshape(-1, 1), t, c)
        hidden = self.model.head[:-1](h)
        v = core.alpha_sigma(t)[1][:, None]*self.model.head[-1](hidden)
        return v.reshape(*shape, 1), hidden.reshape(*shape, -1)

    def forward(self, x, t, c):
        return self.predict_features(x, t, c)[0]


class Ratio(nn.Module):
    def __init__(self, counts, case='bump'):
        super().__init__()
        if counts.ndim != 2 or counts.shape[1] != 64 or 4096 % counts.shape[0]:
            raise ValueError('Condition bins must divide4096; use64 output bins')
        bins = counts.shape[0]
        edges = torch.linspace(-6, 6, 65, dtype=torch.float64)
        edges[0], edges[-1] = -torch.inf, torch.inf
        # Average exact Gaussian bin masses over a fine grid inside each c bin.
        c = ((torch.arange(4096, dtype=torch.float64)+.5)/4096*2-1)[:, None]
        mu = core.mean_function(c, case)
        cdf = torch.special.ndtr((edges[None]-mu)/.35)
        p = (cdf[:, 1:]-cdf[:, :-1]).reshape(bins, 4096//bins, 64).mean(1)
        # Preserve total prior mass across resolutions at fixed sample budget.
        pseudocount = .5*32/bins
        q = (counts.double()+pseudocount)/(counts.sum(-1, keepdim=True)+64*pseudocount)
        self.register_buffer('edges', edges.float())
        self.register_buffer('log_ratio', (p.clamp_min(1e-30).log()-q.log()).float())
        self.register_buffer('counts', counts)

    def ids(self, y, c):
        bins = self.counts.shape[0]
        ci = ((c[..., 0]+1)*bins/2).long().clamp(0, bins-1)
        yi = torch.bucketize(y[..., 0].contiguous(), self.edges[1:-1])
        return ci.expand_as(yi), yi

    def forward(self, y, c, data=None):
        ci, yi = self.ids(y, c)
        return self.log_ratio[ci, yi]


@torch.no_grad()
def sample(model, seed, contexts=2048, candidates=32):
    rng = core.generator(seed)
    c = torch.rand(contexts, 1, generator=rng)*2-1
    noise = torch.randn(contexts, candidates, 1, generator=rng)
    ys = [native.ddim(model, cc[:, None], zz, 20)
          for cc, zz in zip(c.split(128), noise.split(128))]
    y = torch.cat(ys)
    if not torch.isfinite(y).all():
        raise FloatingPointError('Nonfinite DDIM samples')
    return c, y


def run(output, steps=1000, seed=17, case='bump', source_path=None,
        condition_bins=32, calibration_contexts=8192):
    output.mkdir(parents=True, exist_ok=False)
    cfg = placement.Config(case=case)
    data = core.make_dataset(cfg.training, case)
    def emit(row):
        with (output/'progress.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    if source_path is None:
        source, readiness = placement.pretrain(output, data, cfg, seed,
            lambda row: emit({'phase': 'pretrain', 'epoch': row['epoch'],
                              'val_mse': row['validation']['velocity_mse']}))
    else:
        saved = torch.load(source_path, map_location='cpu', weights_only=True)
        if saved['config'] != asdict(cfg) or saved['seed'] != seed:
            raise ValueError('Source configuration/seed must match this experiment')
        source = placement.initialize(cfg, seed)
        source.load_state_dict(saved['model'], strict=True)
        readiness = saved['report']
        core.atomic_checkpoint(output/'source.pt', saved)
    report = {'state': 'running', 'case': case, 'seed': seed, 'steps': steps, 'pretrain': readiness,
              'reward': 'fixed estimated binned truth/source ratio; not exact continuous ratio',
              'velocity_coefficient': 1., 'history': [],
              'source_path': str(source_path) if source_path else None,
              'condition_bins': condition_bins, 'calibration_contexts': calibration_contexts}
    if readiness['status'] != 'ready':
        report['state'] = 'inconclusive_source_not_ready'
        core.atomic_json(output/'report.json', report)
        return
    models = {k: Policy(v) for k, v in placement.fork_models(source).items() if k != 'late'}
    reference = copy.deepcopy(models['none']).eval().requires_grad_(False)
    counts = torch.zeros(condition_bins, 64, dtype=torch.int64)
    dummy = Ratio(counts, case)
    c, y = sample(reference, seed+9000, calibration_contexts, 64)
    ci, yi = dummy.ids(y, c[:, None])
    ids = ci*64+yi
    counts += torch.bincount(ids.flatten(), minlength=condition_bins*64).reshape(condition_bins,64)
    half_counts = [torch.bincount(part.flatten(), minlength=condition_bins*64).reshape(condition_bins,64)
                   for part in ids.chunk(2)]
    critic = Ratio(counts, case).eval()
    core.atomic_checkpoint(output/'reward.pt', critic.state_dict())
    pcfg = native.Config(dimensions=1, context_dim=1, batch=64, candidates=8,
                         timesteps=4, policy_lr=1e-4, policy_steps=steps)
    report['policy_config'] = asdict(pcfg)
    c0, y0 = sample(reference, seed+12000)
    r0 = critic(y0, c0[:, None])
    split_rewards = [Ratio(h, case)(y0,c0[:,None]).double() for h in half_counts]
    a,b = [r-r.mean(-1,keepdim=True) for r in split_rewards]
    split_cosine = float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
    ideal = ((r0.softmax(-1)*r0).sum(-1)-r0.mean(-1)).mean().item()
    report['preflight'] = {'initial_reward': r0.mean().item(),
                          'candidate_tilt_headroom': ideal,
                          'weight_health': native.weight_health(r0),
                          'split_calibration_centered_cosine': split_cosine,
                          'calibration_reliable_at_095': split_cosine >= .95}
    # Exact fork equality on actual generated candidates.
    _, yf = sample(models['early'], seed+12000)
    assert torch.equal(y0, yf)
    optimizers = {k: torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                  lr=pcfg.policy_lr, weight_decay=pcfg.weight_decay) for k,m in models.items()}
    rng = core.generator(seed+14000)
    for step in range(1, steps+1):
        c = torch.rand(64, 1, generator=rng)*2-1
        z = torch.randn(64, 8, 1, generator=rng)
        t = torch.rand(4, 64, generator=rng)*.7
        eps = torch.randn(4, 64, 1, generator=rng)
        diagnostics = {}
        for name, model in models.items():
            loss, stats = native.dgpo_objective(model, reference, critic, None, pcfg,
                c, z, t, eps, velocity_coefficient=1.)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizers[name].step()
            diagnostics[name] = stats
        if step % 100 == 0 or step == steps:
            row = {'phase': 'dgpo', 'step': step, 'arms': {}}
            rewards = {}
            for name, model in models.items():
                c, y = sample(model, seed+12000)
                r = critic(y, c[:, None]); rewards[name] = r
                row['arms'][name] = {'reward_mean': r.mean().item(),
                    'gain': native.paired_gain(r, r0),
                    'weight_health': native.weight_health(r), 'train': diagnostics[name]}
                core.atomic_checkpoint(output/f'{name}_last.pt', {
                    'model': model.state_dict(), 'optimizer': optimizers[name].state_dict(),
                    'step': step, 'rng': rng.get_state(), 'config': asdict(pcfg)})
            row['fourier_minus_none'] = native.paired_gain(rewards['early'], rewards['none'])
            report['history'].append(row)
            core.atomic_json(output/'report.json', report)
            emit(row)
    # Fresh final panel: not used to select the checkpoint or tune this pilot.
    cf, yf = sample(reference, seed+22000, 4096, 32)
    rf = critic(yf, cf[:,None]); final = {}
    for name, model in models.items():
        c, y = sample(model, seed+22000, 4096, 32)
        final[name] = critic(y, c[:,None])
    report['final'] = {name: native.paired_gain(r, rf) for name,r in final.items()}
    report['final']['fourier_minus_none'] = native.paired_gain(final['early'], final['none'])
    margin = .05*ideal
    report['decision'] = {'margin': margin,
        'baseline_stalled_at_tested_budget': report['final']['none']['hi95'] < margin,
        'fourier_material_advantage': report['final']['early']['lo95'] > margin and
                          report['final']['fourier_minus_none']['lo95'] > margin,
        'scope': 'one seed exploratory; binned reward; finite update budget; no EveNet claim'}
    report['decision']['failure_and_rescue'] = (
        split_cosine >= .95 and
        report['decision']['baseline_stalled_at_tested_budget'] and
        report['decision']['fourier_material_advantage'])
    report['state'] = 'completed'
    core.atomic_json(output/'report.json', report)
    emit({'phase': 'completed', 'final': report['final'], 'decision': report['decision']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--case', choices=('bump', 'narrow_bump'), default='bump')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--condition-bins', type=int, choices=(32,128), default=32)
    parser.add_argument('--calibration-contexts', type=int, default=8192)
    args = parser.parse_args()
    if args.steps < 1 or args.calibration_contexts < 256 or args.calibration_contexts % 2:
        parser.error('positive steps and even calibration-contexts >=256 required')
    torch.set_num_threads(2)
    run(args.output, args.steps, case=args.case, source_path=args.source,
        condition_bins=args.condition_bins, calibration_contexts=args.calibration_contexts)
