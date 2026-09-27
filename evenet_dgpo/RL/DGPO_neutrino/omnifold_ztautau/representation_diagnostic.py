"""Detached, fixed-panel representation measurements; never train the judge.

Readouts use early-stop validation only, partitioned by exact packed event
context. They are diagnostic linear ridge scores, not new audit classifiers.
No readout result selects a checkpoint or changes the production optimizer.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import torch

LOG = logging.getLogger(__name__)


def context_partition(condition, byte=0):
    """Same packed event always belongs to the same readout partition/class."""
    rows = condition.detach().cpu().contiguous().numpy()
    return torch.tensor([
        hashlib.sha256(row.tobytes()).digest()[byte] < 128 for row in rows
    ], dtype=torch.bool)


def binary_auc(score, target):
    """Tie-correct AUC without pairwise quadratic storage."""
    _, inverse, counts = torch.unique(score, sorted=True, return_inverse=True, return_counts=True)
    ranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    npos, nneg = int(target.sum()), int((~target).sum())
    return float((ranks[inverse][target].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ridge_readout(features, target, fit_mask, strength=1.0):
    """Balanced ridge regression on +/-1 labels; fixed lambda=1, no tuning.

Standardization is fitted on readout-fit rows only. Evaluate on the other
context-disjoint rows. Invalid/tiny panels emit valid=0, never a fake AUC.
"""
    x = features.detach().cpu().double()
    y, fit = target.cpu().bool(), fit_mask.cpu().bool()
    result = {"valid": 0., "feature_dim": float(x.shape[1])}
    for split, mask in (("fit", fit), ("holdout", ~fit)):
        for label, select in (("positive", y), ("negative", ~y)):
            result[f"{split}_{label}_rows"] = float((mask & select).sum())
    if not torch.isfinite(x).all() or min(v for k, v in result.items() if k.endswith('_rows')) < 8:
        return result
    mean, std = x[fit].mean(0), x[fit].std(0, unbiased=False).clamp_min(1e-6)
    x = torch.cat(((x - mean) / std, torch.ones(len(x), 1, dtype=x.dtype)), dim=1)
    a, labels = x[fit], y[fit].double() * 2 - 1
    weights = torch.where(y[fit], .5 / y[fit].sum(), .5 / (~y[fit]).sum()).double()
    penalty = torch.eye(a.shape[1], dtype=a.dtype) * strength
    penalty[-1, -1] = 0  # Do not penalize the intercept.
    coef = torch.linalg.solve(a.T @ (weights[:, None] * a) + penalty, a.T @ (weights * labels))
    score = x @ coef
    result.update(valid=1., fit_auc=binary_auc(score[fit], y[fit]),
                  holdout_auc=binary_auc(score[~fit], y[~fit]))
    return result


def nested_ridge_readout(features, target, fit_mask, inner_mask):
    """Two-fold context-disjoint CV within outer fit; outer holdout never selects lambda."""
    grid = (1e-4, 1e-2, 1., 100.)
    result = {'valid': 0.}
    x, y, inner = features[fit_mask], target[fit_mask], inner_mask[fit_mask]
    candidates = []
    for strength in grid:
        folds = [ridge_readout(x, y, mask, strength) for mask in (inner, ~inner)]
        if not all(f['valid'] for f in folds):
            return result
        auc = sum(f['holdout_auc'] for f in folds) / 2
        result[f'lambda_{strength:g}/inner_auc'] = auc
        candidates.append((auc, strength))
    # Deterministic ties prefer stronger regularization.
    selected = max(candidates)[1]
    result.update(ridge_readout(features, target, fit_mask, selected))
    result['selected_lambda'] = selected
    return result


class RepresentationDiagnostic:
    def __init__(self, owner, validation):
        self.owner = owner
        self.panel = []
        self.partition = []
        self.inner_partition = []
        for c, z, _ in (validation[:3], validation[3:]):
            index = torch.arange(owner.rank, min(len(z), owner.world * owner.config.representation_probe_rows),
                                 owner.world, device=z.device)
            self.partition.append(context_partition(c[index]))
            self.inner_partition.append(context_partition(c[index], byte=1))
            dtype = next(owner.model.parameters()).dtype
            self.panel.append((c[index].to(device=owner.device, dtype=dtype),
                               z[index].to(device=owner.device, dtype=dtype)))
        self.local_baseline_done = False

    def install_training_hooks(self):
        o = self.owner
        from .stability_diagnostic import diagnostic_modules
        for name, module in diagnostic_modules(o.model).items():
            branch = {"bank.topology_encoder": "fourier", "bank.decoder": "decoder",
                      "bank.topology_standardizer": "fourier_standardized",
                      "bank.decoder.pair_in": "pair_token",
                      "bank.fusion": "fusion"}.get(name)
            if branch:
                def hook(mod, args, output, branch=branch):
                    o.rms(f"representation/{branch}/activation_rms", output)
                    o.watch_gradient(f"representation/{branch}/gradient_rms", output)
                o.handles.append(module.register_forward_hook(hook))
            if name.startswith('bank.decoder.blocks.') and name.endswith('.modulation'):
                def gate_hook(mod, args, output, name=name):
                    for branch, index in (("self", 2), ("cross", 5), ("ffn", 8)):
                        gate = output[index]
                        o.rms(f"representation/{name}/gate_{branch}_rms", gate)
                        o.record(f"representation/{name}/gate_{branch}_absmax", gate.detach().abs().amax())
                        o.record(f"representation/{name}/gate_{branch}_near_zero_fraction", (gate.detach().abs() < 1e-3).float().mean())
                        o.watch_gradient(f"representation/{name}/gate_{branch}_gradient_rms", gate)
                o.handles.append(module.register_forward_hook(gate_hook))

    @torch.no_grad()
    def probe(self, step):
        from .stability_diagnostic import diagnostic_modules, diagnostic_buffers, rng_state, restore_rng
        o = self.owner
        state = rng_state(o.device)
        modules = diagnostic_modules(o.model)
        modes = {name: m.training for name, m in modules.items()}
        buffers = {name: b.detach().clone() for name, b in diagnostic_buffers(o.model).items()}
        hooks, captured, records = [], {}, []
        error = None
        try:
            o.model.eval()
            extended = o.config.representation_path_enabled
            for name, key in (("bank.topology_encoder", "fourier"), ("bank.decoder", "decoder"),
                              ("bank.topology_standardizer", "fourier_standardized"),
                              ("bank.decoder.pair_in", "pair_token"),
                              ("bank.fusion", "fusion")):
                if name in modules:
                    def capture(mod, args, output, key=key):
                        captured[key] = output.detach().reshape(len(output), -1).cpu()
                    hooks.append(modules[name].register_forward_hook(capture))
            if extended and 'bank.topology_encoder.0' in modules:
                def capture_normalization(mod, args, output):
                    captured['raw_fourier'] = args[0].detach().flatten(1).cpu()
                    captured['normalized_fourier'] = output.detach().flatten(1).cpu()
                hooks.append(modules['bank.topology_encoder.0'].register_forward_hook(capture_normalization))
            for label, (c, z), partition, inner in zip((True, False), self.panel, self.partition, self.inner_partition):
                captured.clear()
                logits = o.model(c, z).detach().reshape(-1).cpu()
                records.append((dict(captured), logits, torch.full((len(z),), label), partition, inner))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOG.exception('Representation panel forward failed on rank %s', o.rank)
        finally:
            for hook in hooks:
                hook.remove()
            for name, module in modules.items():
                module.training = modes[name]
            for name, b in diagnostic_buffers(o.model).items():
                b.copy_(buffers[name])
            restore_rng(state, o.device)
        gathered = [(records, error)]
        if o.world > 1:
            gathered = [None] * o.world
            torch.distributed.all_gather_object(gathered, (records, error))
        metrics = {"step": float(step), "error": float(any(e for _, e in gathered))}
        if not metrics["error"]:
            try:
                # Same gathered order on every rank; CPU readouts only on rank 0.
                if o.rank == 0:
                    records = [record for rank_records, _ in gathered for record in rank_records]
                    target = torch.cat([r[2] for r in records]).bool()
                    fit = torch.cat([r[3] for r in records])
                    inner = torch.cat([r[4] for r in records])
                    keys = set.intersection(*(set(r[0]) for r in records))
                    expected = {'decoder'}
                    if 'bank.decoder.pair_in' in modules:
                        expected.add('pair_token')
                    if 'bank.topology_encoder' in modules:
                        expected.add('fourier')
                    if 'bank.fusion' in modules:
                        expected.add('fusion')
                    metrics['branches_complete'] = float(expected <= keys)
                    if o.config.representation_path_enabled:
                        metrics['path_complete'] = float({'raw_fourier', 'normalized_fourier', 'fourier', 'decoder', 'fusion'} <= keys)
                    features = {k: torch.cat([r[0][k] for r in records]) for k in keys}
                    if {'fourier', 'decoder'} <= keys:
                        features['concat'] = torch.cat((features['decoder'], features['fourier']), dim=1)
                    for name, values in features.items():
                        for key, value in ridge_readout(values, target, fit).items():
                            metrics[f"{name}/{key}"] = value
                        if o.config.representation_path_enabled:
                            for key, value in nested_ridge_readout(values, target, fit, inner).items():
                                metrics[f'{name}/cv/{key}'] = value
                    scores = torch.cat([r[1] for r in records])
                    if o.config.representation_export_dir and step in (0, 50, 100):
                        destination = Path(o.config.representation_export_dir)
                        destination.mkdir(parents=True, exist_ok=True)
                        # Exclusive creation: never silently mix restarted fits.
                        with (destination / f'panel_step{step:04d}.pt').open('xb') as stream:
                            torch.save(dict(schema=1, capture_step=int(step), world_size=o.world,
                                            features=features, target=target, fit_mask=fit,
                                            inner_mask=inner, current_logits=scores,
                                            fit_config=__import__('dataclasses').asdict(o.config)), stream)
                        metrics['panel_exported'] = 1.
                    if target.any() and (~target).any() and torch.isfinite(scores).all():
                        metrics['current_head/panel_auc'] = binary_auc(scores, target)
                        for split, mask in (('fit', fit), ('holdout', ~fit)):
                            if target[mask].any() and (~target[mask]).any():
                                metrics[f'current_head/{split}_auc'] = binary_auc(scores[mask], target[mask])
                    metrics['panel_rows'] = float(len(target))
            except Exception:
                LOG.exception('Detached representation readout failed')
                metrics['error'] = 1.
        if o.world > 1:
            payload = [metrics if o.rank == 0 else None]
            torch.distributed.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        prefix = 'stability/representation_probe/' + ('initial/' if step == 0 else '')
        o.metrics.update({prefix + key: value for key, value in metrics.items()})
