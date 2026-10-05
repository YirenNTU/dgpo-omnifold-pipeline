"""Fixed-panel ratio diagnostics. No fitting, candidate generation or selection for deployment."""
import numpy as np

AXES = ('kk', 'kr', 'kn', 'rk', 'rr', 'rn', 'nk', 'nr', 'nn')


def candidate_weights(base, logits):
    """Globally normalized base/K * exp(logit); K cancels for equal draws/event."""
    base, logits = np.asarray(base, float), np.asarray(logits, float)
    if (logits.ndim != 2 or base.shape != (len(logits),) or not np.isfinite(logits).all()
            or not np.isfinite(base).all() or (base < 0).any() or base.sum() <= 0):
        raise ValueError('Invalid base weights or logits')
    logw = np.full_like(logits, -np.inf)
    take = base > 0
    logw[take] = np.log(base[take, None]) + logits[take]
    shift = logw.max()
    weights = np.exp(logw-shift)
    total = weights.sum()
    return weights/total, float(shift+np.log(total)-np.log(base.sum())-np.log(logits.shape[1]))


def arms(logits, cfg):
    yield 'unweighted', np.zeros_like(logits)
    yield 'raw', logits
    for cap in cfg['ratio_caps']:
        yield f'cap_{cap:g}', np.minimum(logits, np.log(cap))
    for alpha in cfg['temperatures']:
        yield f'alpha_{alpha:g}', alpha*logits


def top_indices(values, count):
    values = np.asarray(values).ravel()
    count = min(count, len(values))
    if count < 1:
        return np.empty(0, dtype=int)
    idx = np.argpartition(values, len(values)-count)[-count:]
    return idx[np.argsort(-values[idx], kind='stable')]


def influence_norm(weights, generated, mean):
    # Avoid a full N*K*9 temporary for the production panel.
    norm2 = np.zeros_like(weights)
    for j in range(generated.shape[-1]):
        norm2 += np.square(generated[..., j]-mean[j])
    return weights*np.sqrt(norm2)


def attribution(inputs, data, k, cfg):
    g, s = data['cij'][:, :k], data['logits'][:, :k]
    weights, _ = candidate_weights(inputs['weight'], s)
    mean = np.einsum('nk,nkd->d', weights, g)
    target = np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0)
    strength = influence_norm(weights, g, mean)
    orders = {name: top_indices(value, max(cfg['top_count'], *cfg['remove_counts']))
              for name, value in [('mass', weights), ('influence', strength)]}
    selected = set()
    for order in orders.values():
        selected.update(order[:cfg['top_count']].tolist())
    for j in range(9):
        selected.update(top_indices(weights*np.abs(g[..., j]-mean[j]), cfg['top_count']).tolist())
    records = []
    for flat in sorted(selected, key=lambda v: -strength.ravel()[v]):
        i, draw = divmod(flat, k)
        mass = float(weights[i, draw])
        influence = mass*(g[i, draw]-mean)
        exact_change = None if mass >= 1 else (-influence/(1-mass)).tolist()
        records.append(dict(source_id=str(inputs['source_ids'][i]), event_index=i,
            draw=draw+1, category=int(inputs['category'][i]), visible_pt_sum=float(inputs['visible_pt_sum'][i]),
            base_weight=float(inputs['weight'][i]), kappas=inputs['kappas'][i].tolist(),
            log_ratio=float(s[i, draw]), mass=mass, influence_norm=float(strength[i, draw]),
            cij=g[i, draw].tolist(), weighted_cij_contribution=(mass*g[i, draw]).tolist(),
            influence=influence.tolist(), exact_leave_one_change=exact_change,
            truth_cij=inputs['truth_cij'][i].tolist(), tau=data['tau'][i, draw].tolist(),
            truth_tau=inputs['truth_tau'][i].tolist(), deltas=data['deltas'][i, draw].tolist(),
            visible_a=inputs['visible_a'][i].tolist(), visible_b=inputs['visible_b'][i].tolist()))
    removals = []
    for ranking, order in orders.items():
        for count in cfg['remove_counts']:
            chosen = order[:count]
            ii, jj = np.divmod(chosen, k)
            mass = weights[ii, jj].sum()
            if mass >= 1:
                removals.append(dict(ranking=ranking, removed=len(chosen), valid=False,
                                     reason='No remaining positive weight'))
                continue
            revised = (mean-np.einsum('n,nd->d', weights[ii, jj], g[ii, jj]))/(1-mass)
            removals.append(dict(ranking=ranking, removed=len(chosen), valid=True,
                removed_mass=float(mass), cij=revised.tolist(), error=float(np.linalg.norm(revised-target)),
                matrix_change=(revised-mean).tolist()))
    blocks = []
    for start in range(0, k, 8):
        w = weights[:, start:start+8]
        block_g = g[:, start:start+8]
        contribution = np.einsum('nk,nkd->d', w, block_g)
        blocks.append(dict(first_draw=start+1, last_draw=min(start+8, k), mass=float(w.sum()),
            cij_contribution=contribution.tolist(),
            centered_contribution=(contribution-w.sum()*mean).tolist()))
    return dict(top_candidates=records, removal_sensitivity=removals, draw_blocks=blocks,
        selection='Union of top weight, L2 influence, and per-component absolute influence; post hoc diagnostics only.',
        removal_target='Keep the full original truth target; drop selected generated candidates only, '
                       'then globally renormalize. Biased sensitivity analysis, not a corrected estimator.')


def joint_bins(tau):
    """Fixed physical bins, never optimized using Cij or classifier outcomes."""
    tau = np.asarray(tau)
    def bins(x, edges):
        return np.searchsorted(edges, x, side='right')
    z_edges = [-.5, 0., .5]
    a = np.arctan2(tau[..., 1], tau[..., 0])
    b = np.arctan2(tau[..., 4], tau[..., 3])
    # Map +pi to -pi to respect periodicity.
    a, b = (a+np.pi)%(2*np.pi)-np.pi, (b+np.pi)%(2*np.pi)-np.pi
    return {
        'joint_costheta': (4*bins(tau[..., 2], z_edges)+bins(tau[..., 5], z_edges), 16),
        'joint_phi': (4*bins(a, [-np.pi/2, 0., np.pi/2])+bins(b, [-np.pi/2, 0., np.pi/2]), 16),
        'opening_cos': (bins(np.sum(tau[..., :3]*tau[..., 3:6], axis=-1), np.linspace(-1, 1, 9)[1:-1]), 8),
    }


def clustered_bin_comparison(truth_bins, candidate_bins, base, weights, bins):
    """Paired conditional histogram comparison, clustered by event (not candidate).

    Analytic delta-method SE; finite-sample tails, learned-model uncertainty and
    multiple comparisons are not resolved by these pointwise intervals.
    """
    n, k = candidate_bins.shape
    event = np.arange(n)
    truth_num = np.zeros((n, bins))
    truth_num[event, truth_bins] = base
    gen_num = np.zeros_like(truth_num)
    np.add.at(gen_num, (np.repeat(event, k), candidate_bins.ravel()), weights.ravel())
    truth_mass, gen_mass = base.sum(), weights.sum()
    if truth_mass <= 0 or gen_mass <= 0:
        return None
    truth = truth_num.sum(0)/truth_mass
    gen = gen_num.sum(0)/gen_mass
    influence = (gen_num-weights.sum(1)[:, None]*gen)/gen_mass
    influence -= (truth_num-base[:, None]*truth)/truth_mass
    se = np.sqrt(np.square(influence).sum(0)*n/max(n-1, 1))
    counts = np.bincount(truth_bins[base > 0], minlength=bins)
    gen_counts = np.bincount(candidate_bins[base > 0].ravel(), minlength=bins)
    bin_sq = np.square(gen_num).sum(0)
    bin_ess = np.divide(np.square(gen_num.sum(0)), bin_sq, out=np.zeros(bins), where=bin_sq>0)
    return truth, gen, se, counts, gen_counts, (gen_num>0).sum(0), bin_ess


def closure_rows(inputs, truth_bins, generated_bins, weights, edges, minimum):
    base = inputs['weight']
    ptbin = np.searchsorted(edges, inputs['visible_pt_sum'], side='right')
    groups = [('all', np.arange(len(base)))]
    for category in np.unique(inputs['category']):
        for p in range(len(edges)+1):
            idx = np.flatnonzero((inputs['category']==category)&(ptbin==p))
            if len(idx):
                groups.append((f'category_{category}_pt_{p}', idx))
    rows = []
    for name, idx in groups:
        group_base = base[idx].sum()/base.sum()
        group_weighted = weights[idx].sum()
        for family, (tb, bins) in truth_bins.items():
            gb = generated_bins[family][0]
            result = clustered_bin_comparison(tb[idx], gb[idx], base[idx], weights[idx], bins)
            if result is None:
                if group_base == 0:
                    continue
                raise ValueError('Positive-base condition stratum has zero ratio mass: '+name)
            truth, gen, se, counts, gen_counts, support, bin_ess = result
            for j in range(bins):
                delta = gen[j]-truth[j]
                rows.append(dict(group=name, family=family, bin=j, events=len(idx),
                    truth_count=int(counts[j]), generated_count=int(gen_counts[j]),
                    weighted_support_events=int(support[j]), weighted_bin_event_ess=float(bin_ess[j]),
                    sparse=bool(counts[j]<minimum or bin_ess[j]<minimum), truth_probability=float(truth[j]),
                    generated_probability=float(gen[j]), paired_delta=float(delta),
                    delta_lo95=float(delta-1.96*se[j]), delta_hi95=float(delta+1.96*se[j]),
                    paired_se=float(se[j]), base_group_mass=float(group_base),
                    weighted_group_mass=float(group_weighted),
                    truth_global_mass=float(group_base*truth[j]),
                    generated_global_mass=float(group_weighted*gen[j])))
    return rows


def paired_bootstrap(inputs, nums, dens, bootstrap, seed):
    """Common event resamples for all arms; candidates and paired truth stay together."""
    n = len(inputs['weight'])
    nums, dens = np.stack(nums, 1), np.stack(dens, 1)
    count_rng = np.random.default_rng(seed)
    estimates, targets = [], []
    # Bound memory without storing bootstrap*N counts for the full run.
    for start in range(0, bootstrap, 20):
        count = count_rng.multinomial(n, np.full(n, 1/n), size=min(20, bootstrap-start)).astype(float)
        denominator = np.einsum('bn,na->ba', count, dens)
        truth_den = np.einsum('bn,n->b', count, inputs['weight'])
        if (denominator <= 0).any() or (truth_den <= 0).any():
            raise ValueError('Bootstrap has zero positive mass')
        estimates.append(np.einsum('bn,nad->bad', count, nums)/denominator[..., None])
        targets.append(np.einsum('bn,nd->bd', count, inputs['weight'][:, None]*inputs['truth_cij'])/truth_den[:, None])
    return np.concatenate(estimates), np.concatenate(targets)


def analyze_prefix(inputs, data, k, cfg, pt_edges, progress=None):
    g, s = data['cij'][:, :k], data['logits'][:, :k]
    target = np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0)
    truth_bins, gen_bins = joint_bins(inputs['truth_tau']), joint_bins(data['tau'][:, :k])
    summaries, nums, dens, histograms = [], [], [], {}
    for name, logit in arms(s, cfg):
        weights, log_mean = candidate_weights(inputs['weight'], logit)
        num = np.einsum('nk,nkd->nd', weights, g)
        den = weights.sum(1)
        mean = num.sum(0)
        summaries.append(dict(arm=name, cij=mean.tolist(), error=float(np.linalg.norm(mean-target)),
            candidate_ess=float(1/np.square(weights).sum()), event_ess=float(1/np.square(den).sum()),
            max_candidate_mass=float(weights.max()), log_mean_ratio=log_mean,
            sensitivity_only=name not in ('raw', 'unweighted')))
        nums.append(num); dens.append(den)
        histograms[name] = closure_rows(inputs, truth_bins, gen_bins, weights, pt_edges, cfg['minimum_truth_bin_count'])
        if progress:
            progress(k, name)
    baseline_lookup = {(r['group'], r['family'], r['bin']): r for r in histograms['unweighted']}
    for rows in histograms.values():
        for row in rows:
            baseline = baseline_lookup[(row['group'], row['family'], row['bin'])]
            proposal = baseline['generated_global_mass']
            row['unweighted_global_mass'] = proposal
            row['required_coarse_ratio'] = None if proposal == 0 else row['truth_global_mass']/proposal
            row['applied_coarse_ratio'] = None if proposal == 0 else row['generated_global_mass']/proposal
    estimates, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    errors = np.linalg.norm(estimates-targets[:, None], axis=-1)
    for i, row in enumerate(summaries):
        row['matrix_ci95'] = np.quantile(estimates[:, i], [.025, .975], axis=0).tolist()
        row['error_minus_unweighted'] = row['error']-summaries[0]['error']
        row['error_minus_raw'] = row['error']-summaries[1]['error']
        row['error_minus_unweighted_ci95'] = np.quantile(errors[:, i]-errors[:, 0], [.025, .975]).tolist()
        row['error_minus_raw_ci95'] = np.quantile(errors[:, i]-errors[:, 1], [.025, .975]).tolist()
    attr = attribution(inputs, data, k, cfg)
    lookup = {(r['group'], r['family'], r['bin']): r for r in histograms['raw']}
    for record in attr['top_candidates']:
        ptbin = np.searchsorted(pt_edges, record['visible_pt_sum'], side='right')
        group = f'category_{record["category"]}_pt_{ptbin}'
        cells = joint_bins(np.asarray(record['tau'])[None])
        record['coarse_raw_context'] = {
            family: lookup.get((group, family, int(indices[0])))
            for family, (indices, _) in cells.items()}
    return dict(truth=target.tolist(), arms=summaries, closure=histograms, attribution=attr)
