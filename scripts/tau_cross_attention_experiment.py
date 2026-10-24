"""Warm General plus candidate-query attention; fixed-candidate experiment only."""
import numpy as np

PRIMARY = ('Held-out equal-prior BCE over truth and all K8 candidates: attention must beat '
           'both General and continued General; both upper paired CI95 < 0. '
           'Classifier endpoint only; physics reweighting and concurrence require separate validation.')
ATTENTION_PREFIXES = ('token_encoder.', 'slot_identity.', 'query_norm.',
                      'cross_attention.', 'attention_output.')


def verify_controls(cfg, source):
    """Require finished controls with identical source, fit contract and samples."""
    import json
    from pathlib import Path
    import torch
    from scripts.tau_general_topology_residual import build_arm
    root, control = Path(cfg['source_directory']), Path(cfg['control_directory'])
    if not (control/'COMPLETE').is_file():
        raise ValueError('Control run is not complete')
    previous = json.loads((control/'settings.json').read_text())
    for key in ('source_run','source_directory','policy_checkpoint','policy_step','workers',
                'validation_events','validation_candidates','ratio_cap','seed','fit'):
        if previous.get(key) != cfg.get(key):
            raise ValueError('Control contract differs: '+key)
    if json.loads((control/'wandb.json').read_text())['id'] != cfg['control_run']:
        raise ValueError('Control run ID differs')
    old = torch.load(control/'source_general.pt', map_location='cpu', weights_only=True)
    for key,value in source.items():
        if key == 'state_dict':
            if set(value) != set(old[key]) or any(not torch.equal(v,old[key][k]) for k,v in value.items()):
                raise ValueError('Control General tensors differ')
        elif isinstance(value, (str,int,float,bool,list,dict,type(None))):
            if value != old.get(key):
                raise ValueError('Control General metadata differs: '+key)
    status = json.loads((control/'continued/best.json').read_text())
    if status['total_steps'] != 5500 or status['best_steps'] < cfg['fit']['min_selected_steps']:
        raise ValueError('Incomplete control training budget')
    saved = torch.load(control/'continued/best.pt', map_location='cpu', weights_only=True)
    build_arm(source, 'continued', state=saved['state_dict'])  # strict reload
    with np.load(control/'validation_panel.npz', allow_pickle=False) as p, np.load(root/'validation_panel.npz', allow_pickle=False) as original:
        ids = p['source_ids']
        np.testing.assert_array_equal(ids, original['source_ids'])
        np.testing.assert_array_equal(p['event_weight'], original['event_weight'])
        np.testing.assert_array_equal(p['truth_deltas'], original['truth_deltas'])
    deltas = np.load(control/'validation_deltas.npy', mmap_mode='r')
    for rank in range(16):
        positions = np.arange(rank, cfg['validation_events'], 16)
        with np.load(control/f'scores-rank-{rank:02d}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(f['positions'], positions)
            np.testing.assert_array_equal(f['source_ids'], ids[positions])
            for arm in ('general','continued'):
                for key,shape in ((arm,(len(positions),8)),(arm+'_truth',(len(positions),))):
                    if f[key].shape != shape or not np.isfinite(f[key]).all():
                        raise ValueError('Invalid cached control scores')
        with np.load(root/f'tau_diagnostics/eval-step-00000500/candidates/rank-{rank:02d}.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(deltas[positions], f['deltas'])


def verify_prepared_controls(output, control, *, candidate_prefix=None):
    """Chunked exact equality for fit/selection inputs, excluding old extra features."""
    from pathlib import Path
    for key in ('condition','candidate_truth','candidate_generated','event_weight','split','source_ids'):
        a = np.load(Path(output)/('train_'+key+'.npy'), mmap_mode='r')
        b = np.load(Path(control)/('train_'+key+'.npy'), mmap_mode='r')
        if key.startswith('candidate_'):
            if candidate_prefix is not None:
                if min(a.shape[1], b.shape[1]) < candidate_prefix:
                    raise ValueError('Missing cached General feature prefix')
                a, b = a[:, :candidate_prefix], b[:, :candidate_prefix]
            else:
                b = b[:, :a.shape[1]]
        if a.shape != b.shape:
            raise ValueError('Control training shape differs: '+key)
        for start in range(0,len(a),4096):
            np.testing.assert_array_equal(a[start:start+4096], b[start:start+4096], err_msg=key)
    # All conditioning, weights and reporting inputs must also match externally.
    with np.load(Path(output)/'validation_panel.npz',allow_pickle=False) as a, np.load(Path(control)/'validation_panel.npz',allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Control external panel schema differs')
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def attention_spec():
    return dict(version=1, query='candidate_encoder', tokens='existing_normalized_visible_x',
                width=64, heads=4, slot_embedding=True, zero_output=True,
                placement='before_existing_film_blocks', general_head_trainable=True,
                backbone_trainable=False, extra_candidate_features=0)


def load_general_weights(model, state):
    """Only new attention tensors may be absent; no silent weight dropping."""
    current = model.state_dict()
    added = {k for k in current if k.startswith(ATTENTION_PREFIXES)}
    if not added or set(state) != set(current)-added:
        raise ValueError('General checkpoint keys differ beyond the new attention path')
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != added or result.unexpected_keys:
        raise ValueError('Unexpected partial inheritance')
    import torch
    for key, value in state.items():
        if not torch.equal(current[key], value):
            raise ValueError('General weight not inherited exactly: '+key)


def bce_comparison(positive, negative, weight, bootstrap, seed):
    """Paired whole-event CI; K candidates never count as independent events."""
    arms = ('general', 'continued', 'residual')
    w = np.asarray(weight, dtype=float)
    if bootstrap < 20 or w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid BCE comparison weights/budget')
    losses = []
    shape = None
    for arm in arms:
        p, q = np.asarray(positive[arm], float), np.asarray(negative[arm], float)
        if (p.shape != w.shape or q.ndim != 2 or q.shape[0] != len(w) or q.shape[1] < 1
                or not np.isfinite(p).all() or not np.isfinite(q).all()):
            raise ValueError('Invalid paired truth/candidate scores')
        if shape is not None and q.shape != shape:
            raise ValueError('Unmatched candidate counts')
        shape = q.shape
        losses.append(.5*(np.logaddexp(0, -p)+np.logaddexp(0, q).mean(1)))
    values = np.stack(losses, 1)
    mean = np.average(values, weights=w, axis=0)
    differences = values[:, 2, None]-values[:, :2]
    draws = np.empty((bootstrap, 2))
    rng = np.random.default_rng(seed)
    for j in range(bootstrap):
        counts = np.bincount(rng.integers(0, len(w), len(w)), minlength=len(w))*w
        if counts.sum() <= 0:
            raise ValueError('Empty resampled weight')
        draws[j] = np.einsum('i,ij->j', counts, differences, optimize=False)/counts.sum()
    comparisons = {f'residual_minus_{a}': dict(delta=float(mean[2]-mean[i]),
                        ci95=np.quantile(draws[:, i], [.025, .975]).tolist()) for i,a in enumerate(arms[:2])}
    return dict(primary=PRIMARY, arms={a: dict(bce=float(v)) for a,v in zip(arms,mean)},
                comparisons=comparisons, primary_pass=all(r['ci95'][1] < 0 for r in comparisons.values()),
                events=len(w), candidates=shape[1],
                interpretation='Lower external BCE supports the fitted classifier, not proof of density-ratio or physics closure.',
                uncertainty='Paired whole-event percentile95%; fixed trained heads/candidates; reused external panel is exploratory; no training/systematic uncertainty.')
