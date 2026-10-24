"""Classifier-only warm General control and frozen-General topology correction."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from scripts.tau_topology_branch import TopologyBranchClassifier, topology_spec

ARMS = ('general', 'continued', 'residual')
WARM_CORRECTIONS = ('cross_attention', 'film_mlp')
PAIRS = [('residual', 'general'), ('residual', 'continued'),
         ('residual', 'unweighted'), ('continued', 'general')]
PRIMARY = 'Full-truth Cij distance: residual must beat both General and continued General; both upper paired CI95 < 0'


class FrozenGeneralTopology(TopologyBranchClassifier):
    """One original cap AFTER raw General plus correction; never double-cap."""

    def __init__(self, base, dimension):
        super().__init__(base, topology_spec(dimension), 30)
        self.base.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self


def build_arm(source, arm, *, seed=42, state=None, correction='topology'):
    from scripts.train_conditional_spin_ratio import build_classifier
    if arm not in ARMS:
        raise ValueError('Unknown correction experiment arm')
    if correction not in ('topology', 'fine_angles', 'pair_interaction', *WARM_CORRECTIONS):
        raise ValueError('Unknown correction feature family')
    if (source.get('ratio_bound') != 30 or source.get('dropout') != 0.
            or not source.get('process_label_conditioning')
            or any(source.get(k) for k in ('physics_branch', 'topology_branch', 'explicit_input', 'cross_attention'))):
        raise ValueError('Requires the fitted, label-aware General with cap30 and dropout0')
    cfg = copy.deepcopy(source)
    # Expose the latent only INSIDE the composite model. Its final output still
    # uses exactly the source cap, including at zero correction.
    if arm == 'residual' and correction not in WARM_CORRECTIONS:
        cfg['ratio_bound'] = None
    if arm == 'residual' and correction == 'cross_attention':
        cfg.update(cross_attention=True, attention_heads=4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = build_classifier(cfg)
        if arm == 'residual' and correction == 'cross_attention':
            from scripts.tau_cross_attention_experiment import load_general_weights
            load_general_weights(base, source['state_dict'])
            model = base
        elif arm == 'residual' and correction == 'film_mlp':
            base.load_state_dict(source['state_dict'], strict=True)
            from scripts.tau_film_mlp import add_film_mlp
            model = add_film_mlp(base)
        elif arm == 'residual' and correction == 'pair_interaction':
            base.load_state_dict(source['state_dict'], strict=True)
            from scripts.tau_pair_interaction import FrozenGeneralPairInteraction
            model = FrozenGeneralPairInteraction(base, source['candidate_dim'])
        elif arm == 'residual' and correction == 'fine_angles':
            base.load_state_dict(source['state_dict'], strict=True)
            from scripts.tau_fine_angle_residual import FrozenGeneralFineAngles
            model = FrozenGeneralFineAngles(base, source['candidate_dim'])
        else:
            base.load_state_dict(source['state_dict'], strict=True)
            model = FrozenGeneralTopology(base, source['candidate_dim']) if arm == 'residual' else base
    if state is not None:
        model.load_state_dict(state, strict=True)
    if arm == 'general':
        model.eval().requires_grad_(False)
    return model


def fit_arm(source, arm, fit, arrays, path, device, rank, world, emit, *, seed=42, correction='topology'):
    """Fresh AdamW in both arms; identical fixed update budget, selection by val BCE.

    No test-based selection, warm optimizer inheritance, early stopping or policy
    updates. Source General is already fully fitted before either arm begins.
    """
    from scripts.train_conditional_spin_ratio import score_pair, pair_metrics
    from scripts.tau_residual_ratio import epoch_batches
    def barrier():
        if world > 1:
            dist.barrier()
    if arm not in ('continued', 'residual'):
        raise ValueError('Baseline General must not be trained')
    fi, vi = [np.flatnonzero(arrays['split'] == j) for j in (0, 1)]
    weight = np.asarray(arrays['event_weight'], float)
    if (min(len(fi), len(vi)) < 2 or not np.isfinite(weight).all()
            or (weight < 0).any() or min(weight[fi].sum(), weight[vi].sum()) <= 0):
        raise ValueError('Invalid event-split paired training weights')
    steps_per_epoch = math.ceil(len(fi)/(world*fit['batch_size']))
    expected = fit['epochs']*steps_per_epoch
    if expected < fit['min_selected_steps']:
        raise ValueError('Fixed budget cannot meet minimum selected updates')
    w = weight[fi]/weight[fi].mean()
    bare = build_arm(source, arm, seed=seed, correction=correction).to(device)
    frozen_before = {k: v.detach().cpu().clone() for k, v in bare.base.state_dict().items()} if arm == 'residual' and correction not in WARM_CORRECTIONS else None
    model = torch.nn.parallel.DistributedDataParallel(bare, device_ids=[device.index]
        if device.type == 'cuda' else None) if world > 1 else bare
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=fit['lr'], weight_decay=fit['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, fit['epochs'], eta_min=fit['min_lr'])
    path = Path(path)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
    barrier()
    best_bce, steps = float('inf'), 0
    for epoch in range(fit['epochs']):
        model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        lr = optimizer.param_groups[0]['lr']
        for ii, valid, count in epoch_batches(len(fi), world, fit['batch_size'], rank, seed, epoch):
            idx = fi[ii]
            c, t, g = [torch.as_tensor(np.array(arrays[k][idx]), device=device) for k in
                       ('condition', 'candidate_truth', 'candidate_generated')]
            ew = torch.as_tensor(w[ii]*valid, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            p, q = model(torch.cat((c, c)), torch.cat((t, g))).chunk(2)
            local_sum = (.5*ew*(F.softplus(-p)+F.softplus(q))).sum()
            loss = local_sum*world/count
            finite = torch.isfinite(loss).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('Nonfinite correction BCE')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            totals[0] += local_sum.detach().double()
            totals[1] += valid.sum()
        if world > 1:
            dist.all_reduce(totals)
        local = vi[rank::world]
        p, q = score_pair(bare, arrays['condition'][local], arrays['candidate_truth'][local],
                          arrays['candidate_generated'][local], device, fit['batch_size'])
        gathered = [None]*world if rank == 0 else None
        if world > 1:
            dist.gather_object((local, p, q), gathered, dst=0)
        else:
            gathered = [(local, p, q)]
        if rank == 0:
            ids = np.concatenate([r[0] for r in gathered])
            np.testing.assert_array_equal(np.sort(ids), vi)
            metrics = pair_metrics(np.concatenate([r[1] for r in gathered]),
                                   np.concatenate([r[2] for r in gathered]), weight[ids])
            if steps >= fit['min_selected_steps'] and metrics['bce'] < best_bce:
                best_bce = metrics['bce']
                saved = dict(arm=arm, correction=correction, epoch=epoch+1, optimizer_steps=steps, val_bce=best_bce,
                             state_dict={k: v.detach().cpu().clone() for k, v in bare.state_dict().items()})
                temp = path.with_suffix('.pending.pt')
                torch.save(saved, temp)
                temp.replace(path)
            emit(dict(epoch=epoch+1, optimizer_steps=steps, train_bce=float(totals[0]/totals[1]),
                      val_bce=metrics['bce'], val_auc=metrics['auc'], lr=lr, grad_norm=float(grad),
                      **({'best_val_bce': best_bce} if math.isfinite(best_bce) else {})))
        scheduler.step()
    if steps != expected:
        raise AssertionError('Unmatched optimizer budget')
    barrier()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('correction', 'topology') != correction:
        raise ValueError('Selected correction checkpoint has a different feature family')
    bare.load_state_dict(saved['state_dict'], strict=True)
    if frozen_before is not None:
        for key, value in bare.base.state_dict().items():
            if not torch.equal(value.cpu(), frozen_before[key]):
                raise AssertionError('Frozen General changed: '+key)
    bare.eval().requires_grad_(False)
    status = dict(total_steps=steps, best_steps=saved['optimizer_steps'], best_epoch=saved['epoch'],
                  best_val_bce=saved['val_bce'], fixed_budget=True, optimizer_restarted=True,
                  frozen_general_verified=frozen_before is not None,
                  general_head_trainable=frozen_before is None,
                  trainable_parameters=sum(p.numel() for p in parameters))
    if rank == 0:
        path.with_suffix('.json').write_text(json.dumps(status, indent=2)+'\n')
    return bare, status


def full_truth_comparison(panel, generated, logits, full_truth, bootstrap, seed):
    """Paired events; complete truth is resampled, NOT treated as an exact constant."""
    from scripts.tau_physics_branch_report import terms, errors
    from scripts.tau_tail_attribution import candidate_weights
    w = np.asarray(panel['event_weight'], float)
    n, k = generated.shape[:2]
    t = np.asarray(full_truth, float)
    if t.shape != (n, 9) or not np.isfinite(t).all() or bootstrap < 20:
        raise ValueError('Invalid full-truth moment panel/bootstrap')
    if set(logits) != set(ARMS):
        raise ValueError('Missing comparison arm')
    g = np.stack([terms(panel, generated[:, j]) for j in range(k)], 1)
    scores = {'unweighted': np.zeros((n, k)), **logits}
    names = list(scores)
    columns = [t*w[:, None], w[:, None]]
    matrices = {}
    for name in names:
        s = scores[name]
        if np.shape(s) != (n, k) or np.max(s) > np.log(30)+1e-5:
            raise ValueError('Wrong candidate shape or cap')
        q, _ = candidate_weights(w, s)
        num = np.einsum('nk,nkd->nd', q, g)
        den = q.sum(1)
        columns += [num, den[:, None]]
        matrices[name] = num.sum(0)/den.sum()
    values = np.concatenate(columns, axis=1)
    target = np.average(t, weights=w, axis=0)
    results = {name: dict(C=c.reshape(3, 3).tolist(), **errors(c, target)) for name, c in matrices.items()}
    draws = np.empty((bootstrap, len(PAIRS)))
    rng = np.random.default_rng(seed)
    for b in range(bootstrap):
        ii = rng.integers(0, n, n)
        counts = np.bincount(ii, minlength=n)
        sums = np.einsum('i,ij->j', counts, values, optimize=False).reshape(-1, 10)
        if (sums[:, 9] <= 0).any():
            raise ValueError('Zero bootstrap denominator')
        means = sums[:, :9]/sums[:, 9, None]
        e = dict(zip(names, np.linalg.norm(means[1:]-means[0], axis=1)))
        draws[b] = [e[a]-e[c] for a, c in PAIRS]
    comparisons = {}
    for j, (a, b) in enumerate(PAIRS):
        comparisons[a+'_minus_'+b] = dict(delta=results[a]['error']-results[b]['error'],
                                        ci95=np.quantile(draws[:, j], [.025, .975]).tolist())
    passed = all(comparisons['residual_minus_'+b]['ci95'][1] < 0 for b in ('general', 'continued'))
    return dict(truth_C=target.reshape(3, 3).tolist(), arms=results, comparisons=comparisons,
                primary=PRIMARY, primary_pass=passed,
                interpretation='Exploratory full-truth-reference moment distance, not certified unfolding/concurrence; inspect matched closure, distributions and ESS before any follow-up.',
                uncertainty='Paired whole-event percentile95%; fixed heads/candidates, truth resampled; no training or detector systematic uncertainty.')
