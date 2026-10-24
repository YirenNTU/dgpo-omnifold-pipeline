"""General/Spin/Tau reward lifecycle; native DGPO mathematics unchanged."""
import copy
import json
from pathlib import Path
import numpy as np
import torch

CONTRACT = 'general-spin-tau-epoch287-v1'
ROOT_CONTRACT = 'general-spin-tau-root-exporter-epoch287-v2'


def contract_for_protocol(protocol):
    from scripts.tau_general_spin_angles import PROTOCOL, ROOT_PROTOCOL
    if protocol == PROTOCOL:
        return CONTRACT
    if protocol == ROOT_PROTOCOL:
        return ROOT_CONTRACT
    raise ValueError('Unknown combined reward Spin protocol')


def validate_combined_head(head):
    from scripts.tau_general_spin_angles import branch_spec
    spec = head['general_spin_angles']
    if (head.get('deployment_contract') != contract_for_protocol(spec['spin_protocol'])
            or spec != branch_spec(spec['base_candidate_dim'], spec['spin_protocol'])
            or head['candidate_dim'] != spec['base_candidate_dim']+29
            or head.get('ratio_bound') != 30
            or type(head.get('source_policy_step')) is not int or head['source_policy_step'] < 0
            or head.get('fitting_current_policy') is not True
            or head.get('optimizer_steps', 0) < 1000
            or head.get('dropout') != 0.):
        raise ValueError('Combined head is classifier-only unless explicitly verified for the DGPO reward contract')


def general_audit_inputs(cfg, data):
    """Keep the established fresh, unbounded General audit independent of reward."""
    cfg, data = copy.deepcopy(cfg), dict(data)
    width = cfg.pop('general_spin_angles')['base_candidate_dim']
    cfg.pop('deployment_contract', None)
    cfg['candidate_dim'] = width
    for key in ('candidate_truth', 'candidate_generated'):
        data[key] = data[key][:, :width]
    return cfg, data


def verify_saved_scores(cycle, step=0):
    """Replay selected held-out candidates through the actual production adapter."""
    from .conditional_tau_cycle import panel_batch
    from .conditional_tau_reward import feature_precision
    root = Path(cycle.cfg['combined_classifier_directory'])
    with np.load(root/'scores.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['source_ids'], cycle.validation['source_ids'])
        idx = np.arange(cycle.rank, min(128, len(f['source_ids'])), cycle.world)
        expected = f['log_ratio'][idx, 0]
    draws = np.load(root/'validation_deltas.npy', mmap_mode='r')
    batch = panel_batch(cycle.validation, idx, cycle.reward.spec, cycle.device)
    delta = torch.as_tensor(np.array(draws[idx, 0]), device=cycle.device)[None]
    from scripts.train_conditional_spin_ratio import build_classifier
    from scripts.tau_fresh_negatives import isolated_rng
    with isolated_rng(cycle.device, 42), torch.no_grad(), feature_precision():
        # Always replay the saved initial head, including after a refitted resume.
        head=build_classifier(cycle.reward.bundle['head']).to(cycle.device).eval()
        head.load_state_dict(cycle.reward.bundle['head']['state_dict'],strict=True)
        c,f=cycle.reward.features(batch,delta[0])
        actual=head(torch.as_tensor(c,device=cycle.device),torch.as_tensor(f,device=cycle.device)).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-3)
    # The positive path is different in the corrected protocol. Replay it too,
    # including saved scores, so a stale same-width cache cannot pass startup.
    from scripts.tau_general_spin_angles import ROOT_PROTOCOL, truth_features
    spec = cycle.reward.bundle['head']['general_spin_angles']
    if spec['spin_protocol'] == ROOT_PROTOCOL:
        with np.load(root/'scores.npz', allow_pickle=False) as saved:
            expected_truth = saved['truth_logits'][idx]
        b = panel_batch(cycle.validation, idx, cycle.reward.spec, cycle.device)
        with torch.no_grad(), feature_precision():
            c, f = cycle.reward.features(b, torch.as_tensor(cycle.validation['truth_deltas'][idx], device=cycle.device))
            f[:, spec['base_candidate_dim']:] = truth_features(cycle.validation, idx)
            np.testing.assert_allclose(f, cycle.validation['candidate_truth'][idx], rtol=1e-4, atol=1e-3)
            actual_truth = head(torch.as_tensor(c,device=cycle.device),torch.as_tensor(f,device=cycle.device)).cpu().numpy()
        np.testing.assert_allclose(actual_truth, expected_truth, rtol=1e-4, atol=1e-3)
    if cycle.rank == 0:
        cycle.emit({'tau/startup/combined_saved_score_replay': 1}, step)


def native_cij_monitor(panel, draws, baseline, *, bootstrap=1000, seed=42):
    """Paired event bootstrap of native samples, never reward-weighted samples."""
    from scripts.run_tau_general_spin_angles import tihsu_panel
    from scripts.tau_tihsu_cij import panel_report
    from scripts.tau_general_spin_angles import target_spin_observables, prediction_spin_observables
    from scripts.tau_general_spin_angles_report import products, bin_contributions
    panel = tihsu_panel(panel)
    if draws.shape != baseline.shape or draws.shape[0] != len(panel['source_ids']):
        raise ValueError('Paired Cij comparison requires aligned identical K panels')
    report = panel_report(panel, draws)
    t, tv = bin_contributions(products(target_spin_observables(panel)))
    bank = {}; errors = {'baseline': 0., 'current': 0.}
    for ch in np.unique(panel['category']):
        take = panel['category'] == ch; w = np.asarray(panel['event_weight'][take], float)
        mass = float(w.sum()/np.sum(panel['event_weight']))
        coefficient = 9/np.prod(panel['kappas'][take][0])
        pairs = {'target': (w[:, None]*t[take], w[:, None]*tv[take])}
        for key, d in (('baseline', baseline), ('current', draws)):
            g, gv = bin_contributions(products(prediction_spin_observables(panel, d[take], take)))
            pairs[key] = (w[:, None]*g.mean(1), w[:, None]*gv.mean(1))
        means = {key: coefficient*num.sum(0)/den.sum(0) for key, (num, den) in pairs.items()}
        for key in errors:
            errors[key] += mass*float(np.linalg.norm(means[key]-means['target']))
        bank[str(ch)] = (pairs, coefficient, mass, len(w))
    rng = np.random.default_rng(seed); changes = []
    for _ in range(bootstrap):
        delta = 0.
        for pairs, coefficient, mass, n in bank.values():
            idx = rng.integers(0, n, n)
            means = {}
            for key, (num, den) in pairs.items():
                denom = den[idx].sum(0)
                if np.any(denom <= 0): raise ValueError('Empty bootstrap channel histogram')
                means[key] = coefficient*num[idx].sum(0)/denom
            delta += mass*(np.linalg.norm(means['current']-means['target'])-
                           np.linalg.norm(means['baseline']-means['target']))
        changes.append(delta)
    report['native_change'] = dict(errors, delta=errors['current']-errors['baseline'],
        ci95=np.quantile(changes, [.025, .975]).tolist(), bootstrap=bootstrap,
        scope='Unweighted policy samples; paired whole-event, channel-stratified bootstrap versus step0; no unfolding')
    return report
