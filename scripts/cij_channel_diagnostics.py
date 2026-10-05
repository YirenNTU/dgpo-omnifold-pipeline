"""Channel-resolved closure and exact descriptive matrix decomposition."""
import numpy as np

from scripts.diagnose_reweighted_cij import analyze, prepare


def angular_channel_report(data, mode, bootstrap, seed, references):
    """Direct <a_i b_j>, not 9<a_i b_j> and not inverse-kappa moments."""
    source = analyze(data, [1, 1], mode, bootstrap, seed, references=references)
    def convert(report):
        results = {}
        for arm, values in report['results'].items():
            results[arm] = {
                new: (np.asarray(values[old]) / 9).tolist()
                for old, new in [('C', 'moment'), ('C_ci95', 'moment_ci95'),
                    ('C_minus_truth', 'minus_target'),
                    ('C_minus_truth_ci95', 'minus_target_ci95'),
                    ('frobenius_error', 'frobenius_error')]}
        comparison = report['comparison']
        return dict(results=results,
            error_change={key: (np.asarray(value)/9).tolist() for key,value in
                comparison['C_frobenius_error_change'].items()},
            absolute_error_change=(np.asarray(comparison['absolute_error_change_matrix'])/9).tolist(),
            absolute_error_change_ci95=(np.asarray(comparison['absolute_error_change_matrix_ci95'])/9).tolist())
    return dict(definition='M_ij = weighted mean(a_i*b_j); no factor 9, no analyzing powers',
        axes=source['axes'], bootstrap=bootstrap, seed=seed,
        uncertainty=source['interpretation'],
        references={'stored_truth': convert(source), **{
            name: convert(report) for name,report in source['reference_comparisons'].items()}})


def channel_diagnostics(data, kappas, categories, mode, bootstrap=1000, seed=42, references=None):
    categories = np.asarray(categories)
    n = len(data['event_id'])
    if categories.shape != (n,) or categories.dtype.kind not in 'iu':
        raise ValueError('event_category must be one integer per aligned event')
    references = references or {}
    num, den, _ = prepare(data, kappas, mode)
    total_den = den.sum(0)
    groups = []
    fixed = np.zeros(9)
    total_base = total_den[1]
    if total_base <= 0 or total_den[2] <= 0:
        raise ValueError('No positive total weight')
    undefined_fixed = []
    for category in np.unique(categories):
        mask = categories == category
        d = den[mask].sum(0)
        base_fraction, weighted_fraction = float(d[1]/total_base), float(d[2]/total_den[2])
        group = dict(event_category=int(category), events=int(mask.sum()),
                     base_fraction=base_fraction, reweighted_fraction=weighted_fraction,
                     fraction_change=weighted_fraction-base_fraction)
        if d[1] <= 0 or d[2] <= 0:
            group['status'] = 'zero_total_weight; channel closure unavailable'
            if d[1] > 0:
                undefined_fixed.append(int(category))
            groups.append(group)
            continue
        fixed += base_fraction * num[mask, 2].sum(0)/d[2]
        if mask.sum() < 2:
            group['status'] = 'fewer_than_two_events; bootstrap unavailable'
            groups.append(group)
            continue
        subset = {key: np.asarray(value)[mask] for key, value in data.items()}
        powers = np.asarray(kappas)
        if powers.ndim == 2:
            powers = powers[mask]
        refs = {key: (np.asarray(a)[mask], np.asarray(b)[mask]) for key, (a,b) in references.items()}
        group['report'] = analyze(subset, powers, mode, bootstrap, seed, references=refs)
        group['angular_moments'] = angular_channel_report(subset, mode, bootstrap, seed, refs)
        group['status'] = 'computed'
        groups.append(group)
    overall = num.sum(0)/total_den[:,None]
    result = dict(channels=groups, events=n, weight_mode=mode,
        scope='Same events/candidates/weights. No fitting, generation, cuts, or weight changes.',
        uncertainty='Channel intervals are paired event bootstrap conditional on fitted models; pointwise, not multiplicity corrected. Decomposition is point estimate only.')
    if undefined_fixed:
        result['decomposition'] = dict(available=False, zero_weight_channels=undefined_fixed)
        return result
    within = fixed-overall[1]
    mixture = overall[2]-fixed
    errors = {}
    targets = {'stored_truth': overall[0]}
    # Recover alternative target matrices with the same base-event measure.
    from scripts.diagnose_reweighted_cij import features
    ew = np.asarray(data['event_weight'])
    for name, (a,b) in references.items():
        targets[name] = (ew[:,None]*features(np.asarray(a),np.asarray(b),kappas)).sum(0)/ew.sum()
    for name, target in targets.items():
        errors[name] = {label: float(np.linalg.norm(matrix-target)) for label,matrix in
                       [('unweighted',overall[1]),('reweighted',overall[2]),('fixed_channel_mix',fixed)]}
    result['decomposition'] = dict(available=True,
        unweighted_C=overall[1].reshape(3,3).tolist(), reweighted_C=overall[2].reshape(3,3).tolist(),
        fixed_channel_mix_C=fixed.reshape(3,3).tolist(),
        within_channel_shift=within.reshape(3,3).tolist(),
        mixture_shift=mixture.reshape(3,3).tolist(),
        reconstruction_error=float(np.max(np.abs(within+mixture-(overall[2]-overall[1])))),
        errors=errors,
        definition='fixed_channel_mix = sum(base channel fraction * reweighted within-channel C). Signed shifts add exactly; error norms do not. Descriptive, not causal; not a proposed production weight correction.')
    return result
