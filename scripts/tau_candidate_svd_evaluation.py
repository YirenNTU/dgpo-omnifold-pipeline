"""Actual RooUnfoldSvd with event-level covariance and paired response resampling.

This evaluator never substitutes an unregularized inverse for the SVD solver.
Shared cached-candidate evaluator copied from the verified toy SVD engine.
Legacy Invert reports and their original implementation remain unchanged.
"""
from __future__ import annotations

import itertools
import os
import numpy as np
from scipy.stats import chi2

LIBRARY = os.environ.get('ROOUNFOLD_LIBRARY', '/Users/yirenwu/EveNet-private/RooUnfold/build-myeve/libRooUnfold.dylib')
IDS = itertools.count()


def load_root():
    import ROOT as rt
    rt.gROOT.SetBatch(True)
    if rt.gSystem.Load(LIBRARY) < 0:
        raise RuntimeError('RooUnfold library could not be loaded')
    return rt


class SVD:
    def __init__(self, matrix, kreg):
        rt = load_root(); self.rt = rt
        matrix = np.asarray(matrix, dtype=float)
        self.n = n = len(matrix); self.kreg = int(kreg)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError('Require a finite nonnegative square response')
        if (matrix.sum(0) <= 0).any() or not 1 <= kreg <= n or kreg != self.kreg:
            raise ValueError('Unsupported truth bin or invalid kreg')
        self.matrix = matrix
        name = f'tauSvd{next(IDS)}'
        r = rt.TH1D(name+'r', '', n, 0, n); t = rt.TH1D(name+'t', '', n, 0, n)
        m = rt.TH2D(name+'m', '', n, 0, n, n, 0, n); d = rt.TH1D(name+'d', '', n, 0, n)
        for h in (r, t, m, d): h.SetDirectory(0); h.Sumw2()
        for i in range(n):
            for h, value in ((r, matrix[i].sum()), (t, matrix[:, i].sum())):
                h.SetBinContent(i+1, float(value)); h.SetBinError(i+1, float(np.sqrt(value)))
            for j in range(n):
                m.SetBinContent(i+1, j+1, float(matrix[i, j]))
                # Internal response errors are disabled. Event bootstraps below
                # supply their covariance, including fractional-candidate effects.
                m.SetBinError(i+1, j+1, float(np.sqrt(matrix[i, j])))
        self.keep = r, t, m, d
        self.response = rt.RooUnfoldResponse(r, t, m)

    def solve(self, measured, covariance):
        rt = self.rt; n = self.n; d = self.keep[-1]
        measured = np.asarray(measured, dtype=float); covariance = np.asarray(covariance, dtype=float)
        if measured.shape != (n,) or covariance.shape != (n, n):
            raise ValueError('Measured shape/covariance mismatch')
        if not np.isfinite(measured).all() or not np.isfinite(covariance).all():
            raise ValueError('Nonfinite input')
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-9)
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError('Measured covariance must be positive definite')
        rc = rt.TMatrixD(n, n)
        for i in range(n):
            d.SetBinContent(i+1, float(measured[i])); d.SetBinError(i+1, float(np.sqrt(covariance[i, i])))
            for j in range(n): rc[i][j] = float(covariance[i, j])
        solver = rt.RooUnfoldSvd(self.response, d, self.kreg)
        solver.SetVerbose(0); solver.IncludeSystematics(0); solver.SetMeasuredCov(rc)
        h = solver.Hunfold(2); ec = solver.Eunfold(2)
        value = np.array([h.GetBinContent(i+1) for i in range(n)])
        cov = np.array([[ec[i][j] for j in range(n)] for i in range(n)])
        if not np.isfinite(value).all() or not np.isfinite(cov).all() or value.sum() <= 0:
            raise ValueError('Invalid SVD output')
        # Cancellation can leave O(1e-6) asymmetry in near-zero off-diagonals
        # when variances are O(1e5). Test at the covariance scale, then remove
        # numerical antisymmetry; never accept a materially asymmetric matrix.
        scale = max(float(np.linalg.norm(cov)), 1.)
        if np.linalg.norm(cov-cov.T)/scale > 1e-8:
            raise ValueError('Materially asymmetric RooUnfold covariance')
        cov = (cov+cov.T)/2
        if np.linalg.eigvalsh(cov).min() < -1e-8*scale:
            raise ValueError('Non-positive RooUnfold covariance')
        np.testing.assert_allclose(np.diag(cov), [h.GetBinError(i+1)**2 for i in range(n)], rtol=1e-7, atol=1e-8)
        return value, cov


def shape(value, covariance):
    p = value/value.sum(); jac = (np.eye(len(value))-p[:, None])/value.sum()
    return p, jac@covariance@jac.T


def response(h, truth):
    return np.stack([h[truth == i].sum(0) for i in range(h.shape[1])], axis=1)


def closure(value, covariance, reference, reference_covariance):
    p, pc = shape(value, covariance); delta = p-reference; c = pc+reference_covariance
    statistic = float(delta[:-1]@np.linalg.solve(c[:-1, :-1], delta[:-1]))
    return dict(probability=p.tolist(), max_abs_pull=float(np.max(abs(delta)/np.sqrt(np.diag(c)))),
                bin_pulls=(delta/np.sqrt(np.diag(c))).tolist(), chi2=statistic, ndof=len(p)-1,
                pvalue=float(chi2.sf(statistic, len(p)-1)))


def finite_difference_check(engine, y, cov):
    """Check ROOT error propagation with fixed whitening, including off-diagonals."""
    _, actual = engine.solve(y, cov); jac = np.empty_like(cov)
    step = .01
    for j in range(len(y)):
        delta = np.zeros_like(y); delta[j] = step
        jac[:, j] = (engine.solve(y+delta, cov)[0]-engine.solve(y-delta, cov)[0])/(2*step)
    expected = jac@cov@jac.T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-5)
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(actual))


def evaluate(hs, ht, truth, truth_test, reference, edges, out, *, names, kreg=5, reps=300, seed=124301, pairs=None):
    """All replicas and shifted-truth checks use actual RooUnfoldSvd calls.

    h[event,bin] sums to one: K candidates are correlated contributions of one
    event. Data covariance is compound-Poisson H.T@H. Response is resampled by
    event with fixed response-MC size, paired across all compared arms.
    """
    nb = len(edges)-1; n = len(truth_test); nm = len(names)
    if len(hs) != nm or len(ht) != nm or reps < 2: raise ValueError('Invalid arm count/replicates')
    if pairs is None:
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (3, 2)] if nm == 4 else [(0, i) for i in range(1, nm)]
    if any(not (0 <= a < nm and 0 <= b < nm and a != b) for a, b in pairs):
        raise ValueError('Invalid comparison indices')
    for h, count in [(h, len(truth)) for h in hs]+[(h, n) for h in ht]:
        if h.shape != (count, nb) or not np.isfinite(h).all() or (h < 0).any(): raise ValueError('Invalid event contributions')
        np.testing.assert_allclose(h.sum(1), 1, rtol=0, atol=1e-6)
    pref = np.bincount(reference, minlength=nb)/len(reference)
    if (pref <= 0).any(): raise ValueError('Reference bins must have support')
    refcov = (np.diag(pref)-np.outer(pref, pref))/len(reference)
    expected = n*pref; engines = [SVD(response(h, truth), kreg) for h in hs]
    # Nominal + two smooth truth tilts. Reweight measured events only; each
    # original response (and its bootstrap replicas) stays fixed for the check.
    factors = [np.ones(nb), np.exp(-.3*np.linspace(-1, 1, nb)), np.exp(.3*np.linspace(-1, 1, nb))]
    qs = []; qcs = []; sets = []
    for f in factors:
        norm = pref@f; q = pref*f/norm
        jac = (np.diag(f)-q[:, None]*f[None, :])/norm
        qs.append(q); qcs.append(jac@refcov@jac.T)
        sets.append([h*(q/pref)[truth_test, None] for h in ht])
    observations = [[(h.sum(0), h.T@h) for h in group] for group in sets]
    nominal = [[e.solve(y, c) for e, (y, c) in zip(engines, group)] for group in observations]
    checks = [finite_difference_check(e, *obs) for e, obs in zip(engines, observations[0])]
    draws = np.empty((reps, len(sets), nm, nb)); data_sigma = np.empty((reps, nm))
    dist = np.empty((reps, nm)); poisson = np.empty((reps, nm, nb))
    rng = np.random.default_rng(seed)
    for rep in range(reps):
        rw = rng.multinomial(len(truth), np.full(len(truth), 1/len(truth)))
        mw = rng.multinomial(n, np.full(n, 1/n)); pw = rng.poisson(1, n)
        for m, (hr, hm, engine) in enumerate(zip(hs, ht, engines)):
            er = SVD(response(hr*rw[:, None], truth), kreg)
            for s, group in enumerate(observations): draws[rep, s, m] = er.solve(*group[m])[0]
            v, c = er.solve(mw@hm, hm.T@(mw[:, None]*hm))
            data_sigma[rep, m] = np.sqrt(np.trace(shape(v, c)[1]))
            dist[rep, m] = abs(mw@hm/n-pref).sum()/2
            # Fixed covariance/whitening: validates conditional propagation,
            # not coverage against an alternative physical truth spectrum.
            poisson[rep, m] = engine.solve(pw@hm, observations[0][m][1])[0]
        if (rep+1) % 50 == 0: print('SVD paired response replicas', rep+1, flush=True)
    rows = []; shifted = []; arrays = dict(response_draws=draws, data_sigma_draws=data_sigma,
        tv_draws=dist, poisson_toys=poisson, reference_probability=pref, edges=edges,
        response_truth=truth, measurement_truth=truth_test)
    for m, (name, hm) in enumerate(zip(names, ht)):
        v, c = nominal[0][m]; rc = np.cov(draws[:, 0, m], rowvar=False); total = c+rc
        p, pc = shape(v, total); _, dc = shape(v, c); _, rsc = shape(v, rc)
        centered = draws[:, 0, m]-draws[:, 0, m].mean(0)
        contribution = np.sum(centered**2, axis=1)
        tail_fraction = float(contribution.max()/contribution.sum())
        row = dict(name=name, distribution=hm.mean(0).tolist(), tv=float(abs(hm.mean(0)-pref).sum()/2),
            data_shape_rss=float(np.sqrt(np.trace(dc))), response_shape_rss=float(np.sqrt(np.trace(rsc))),
            total_shape_rss=float(np.sqrt(np.trace(pc))), response_variance_fraction=float(np.trace(rsc)/np.trace(pc)),
            count_sigma_over_poisson=(np.sqrt(np.diag(total))/np.sqrt(expected)).tolist(),
            data_count_sigma_over_poisson=(np.sqrt(np.diag(c))/np.sqrt(expected)).tolist(),
            unfolded_probability=p.tolist(), negative_bins=int((v < 0).sum()),
            closure=closure(v, total, pref, refcov), valid_bootstrap_replicas=reps,
            largest_replica_variance_fraction=tail_fraction, response_tail_warning=tail_fraction > .25,
            poisson_sigma_over_analytic=(poisson[:, m].std(0, ddof=1)/np.sqrt(np.diag(c))).tolist())
        rows.append(row)
        for suffix, arr in [('counts', v), ('data_covariance', c), ('response_covariance', rc),
                            ('shape_covariance', pc), ('response_matrix', engines[m].matrix),
                            ('response_h', hs[m]), ('measured_h', hm)]: arrays[f'model{m}_{suffix}'] = arr
        for s, tilt in [(1, -.3), (2, .3)]:
            sv, sc = nominal[s][m]; sc = sc+np.cov(draws[:, s, m], rowvar=False)
            shifted.append(dict(model=name, tilt=tilt, **closure(sv, sc, qs[s], qcs[s])))
    comparisons = {}
    for a, b in pairs:
        comparisons[names[b]+' / '+names[a]] = dict(
            total_ratio=rows[b]['total_shape_rss']/rows[a]['total_shape_rss'],
            data_ratio=rows[b]['data_shape_rss']/rows[a]['data_shape_rss'],
            data_ratio_ci95=np.quantile(data_sigma[:, b]/data_sigma[:, a], [.025, .975]).tolist(),
            tv_gain_ci95=np.quantile(dist[:, a]-dist[:, b], [.025, .975]).tolist())
    np.savez_compressed(out/'unfolding_arrays.npz', **arrays)
    return dict(method='RooUnfoldSvd', kreg=kreg, models=rows, comparisons=comparisons,
                shifted_closure=shifted, replicates=reps, seed=seed,
                fixed_whitening_covariance_check_relative_errors=checks,
                interval_scope='Paired event bootstrap interval for data-only sigma ratio; not a total-sigma ratio interval.',
                regularization_scope='Common preset kreg; closure tested, not optimized. Smoothing is in bin index, with unequal physical widths.')
