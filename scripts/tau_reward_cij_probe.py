"""Fixed-candidate Cij counterfactuals; no policy or reward updates."""
import numpy as np


def compare_reweighting(truth, candidates, log_ratio, event_weight):
    truth, candidates, log_ratio, w = map(np.asarray, (truth, candidates, log_ratio, event_weight))
    n, k, d = candidates.shape
    if (d != 9 or truth.shape != (n, 9) or log_ratio.shape != (n, k)
            or w.shape != (n,) or k < 2 or any(not np.isfinite(x).all() for x in (truth, candidates, log_ratio, w))
            or (w < 0).any() or w.sum() <= 0):
        raise ValueError('Invalid fixed-candidate Cij panel')
    # Native reward already includes training-time bound; no new cap/temperature.
    ratios = np.exp(log_ratio - log_ratio.max())
    local = np.exp(log_ratio - log_ratio.max(axis=1, keepdims=True))
    local /= local.sum(axis=1, keepdims=True)
    weights = dict(unweighted=np.broadcast_to(w[:, None]/k, (n, k)),
                   global_ratio=w[:, None]*ratios/k,
                   within_condition=w[:, None]*local)
    target = np.average(truth, weights=w, axis=0)
    report = dict(events=n, candidates=k, truth=target.reshape(3, 3).tolist(), arms={})
    for name, mass in weights.items():
        mass = mass/mass.sum()
        value = np.einsum('nk,nkd->d', mass, candidates)
        delta = (value-target).reshape(3, 3)
        report['arms'][name] = dict(cij=value.reshape(3, 3).tolist(),
            error=float(np.linalg.norm(delta)), diagonal_error=float(np.linalg.norm(np.diag(delta))),
            offdiagonal_error=float(np.linalg.norm(delta[~np.eye(3, dtype=bool)])),
            nn_error=float(abs(delta[2, 2])), candidate_ess_fraction=float(1/(n*k*np.sum(mass**2))),
            condition_mass_tv=float(.5*np.abs(mass.sum(axis=1)-w/w.sum()).sum()))
    report['within_condition_ess_fraction'] = float(np.average(1/(k*np.sum(local**2, axis=1)), weights=w))
    return report
