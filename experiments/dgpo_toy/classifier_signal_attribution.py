"""Toy-only frozen-judge attribution; no fitting, policy updates or oracle inputs.

Counterfactual mode/shape swaps describe the SAVED judge, not a best response
trained on the swapped population. Walsh decomposition is exact on the eight
sign modes; within-mode means are Monte Carlo estimates. See the saved plan.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import conditional as native
from . import film_conditioning as film
from .closed_loop_lab import FeatureCritic, HISTORY, load_source, predictions, score_metrics, toy_path
from .cube_swap import modes, swaps, truth_shape
from .nonperiodic_cube import Data
from .truth_pretrain import atomic_checkpoint, atomic_json


def walsh(centers):
    """Columns: constant, x/y/z, xy/xz/yz, xyz. H.T H = 8 I."""
    subsets = [s for k in range(4) for s in combinations(range(3), k)]
    h = torch.stack([centers[:, list(s)].prod(-1) for s in subsets], -1).double()
    return h, torch.tensor([len(s) for s in subsets])


def probabilities(ids):
    return F.one_hot(ids, 8).double().mean(1)


def structure(pool, centers):
    """Exact order split and independent-half correction of histogram noise."""
    y, ids, p = pool['y'].double(), pool['ids'], pool['p'].double()
    h, orders = walsh(centers)
    q = probabilities(ids)
    d = (q-p) @ h
    half = ids.shape[1]//2
    left, right = (probabilities(ids[:, sl])-p for sl in (slice(None,half), slice(half,None)))
    cross = (left @ h)*(right @ h)
    result = {}
    for order in (1, 2, 3):
        mask = orders == order
        result[f'order{order}_mode_l2'] = d[:,mask].square().sum(-1)/8
        # Can be negative from Monte Carlo noise. Never clamp this estimator.
        result[f'order{order}_mode_l2_debiased'] = cross[:,mask].sum(-1)/8
    result['mode_l2'] = (q-p).square().sum(-1)
    residual = result['mode_l2']-sum(result[f'order{k}_mode_l2'] for k in (1,2,3))
    if residual.abs().max() > 1e-10:
        raise AssertionError('Walsh probability-error identity failed')
    result['mode_tv'] = .5*(q-p).abs().sum(-1)
    eps = y-centers[ids].double()
    biases, covs = [], []
    for g in range(len(y)):
        cell_bias, cell_cov = [], []
        for m in range(8):
            e = eps[g,ids[g]==m]
            if len(e) < 16:
                raise ValueError('Insufficient mode count for shape diagnostics')
            cell_bias.append(e.mean(0).square().sum())
            cov = torch.cov(e.T)
            cell_cov.append((cov-torch.eye(3)*.15**2).square().sum())
        biases.append(torch.stack(cell_bias).mean())
        covs.append(torch.stack(cell_cov).mean())
    result['shape_centroid_mse_uniform_modes'] = torch.stack(biases)
    result['shape_covariance_frobenius2_uniform_modes'] = torch.stack(covs)
    result['within_mode_rms'] = eps.square().sum(-1).mean(-1).sqrt()
    return result


def score_decomposition(p, q, truth_means, gen_means, centers):
    """E_truth[r] - E_generator[r], anchored at truth within-mode shape.

    Signed order contributions can cancel. This is not a KL decomposition or
    a causal allocation of why the classifier learned its function.
    """
    p,q,truth_means,gen_means = [v.double() for v in (p,q,truth_means,gen_means)]
    if not all(torch.isfinite(v).all() for v in (p,q,truth_means,gen_means)):
        raise FloatingPointError('Nonfinite score decomposition input')
    if any((v<0).any() or (v.sum(-1)-1).abs().max()>1e-6 for v in (p,q)):
        raise ValueError('Invalid mode probabilities')
    h, orders = walsh(centers)
    coefficients = truth_means @ h / 8
    moment_difference = (p-q) @ h
    result = {f'order{k}_mode_logit_gap':
              (coefficients[:,orders==k]*moment_difference[:,orders==k]).sum(-1)
              for k in (1,2,3)}
    result['within_mode_shape_logit_gap'] = (q*(truth_means-gen_means)).sum(-1)
    reconstructed = sum(result.values())
    result['total_logit_gap'] = (p*truth_means-q*gen_means).sum(-1)
    result['identity_residual'] = result['total_logit_gap']-reconstructed
    if result['identity_residual'].abs().max() > 1e-6:
        raise AssertionError('Mode/shape score identity failed')
    return result


@torch.no_grad()
def cell_means(model, pool, truth_y):
    c,y,ids = pool['c'],pool['y'],pool['ids']
    def infer(yy):
        cc = c[:,None].expand(-1,yy.shape[1],-1).reshape(-1,1)
        return torch.cat([model(a,b) for a,b in zip(yy.reshape(-1,3).split(1024),cc.split(1024))]).reshape(yy.shape[:2]).double()
    g = infer(y)
    means = torch.stack([torch.stack([g[i,ids[i]==m].mean() for m in range(8)]) for i in range(len(c))])
    t = infer(truth_y.reshape(len(c),-1,3)).reshape(truth_y.shape[:3]).mean(-1)
    return t, means


def bce_by_condition(scores, grid):
    return .5*(F.softplus(-scores['positive'].double()) +
               F.softplus(scores['negative'].double())).reshape(grid,-1).mean(-1)


def simultaneous_mean_intervals(series, repeats, seed):
    """Paired resampling of CONDITION nodes, never pretend within-node draws iid.

    A fixed grid is quadrature, not an iid condition sample. These are descriptive
    condition-resampling intervals conditional on fitted models/generated pools;
    not exact population confidence bounds or training-seed uncertainty.
    """
    names = list(series)
    values = np.stack([series[n].double().numpy() for n in names],1)
    point = values.mean(0)
    rng = np.random.default_rng(seed)
    draws = np.stack([values[rng.integers(len(values),size=len(values))].mean(0) for _ in range(repeats)])
    # A common max-deviation radius controls this predeclared metric family.
    radius = float(np.quantile(np.abs(draws-point).max(1),.95))
    return {n:{'mean':float(v),'lo95':float(v-radius),'hi95':float(v+radius)} for n,v in zip(names,point)}


def load_judge(campaign, step, feature):
    folder = campaign/('round03' if feature=='plain' else 'round02')/'classifiers'
    name = f'step{step}__{feature}'
    fit = json.loads((folder/'fit_report.json').read_text())
    if not fit['test'][name]['valid']:
        raise ValueError(f'Inadequate saved fit: {name}')
    path = folder/f'{name}_best.pt'
    ck = torch.load(path,map_location='cpu',weights_only=True)
    if ck['feature'] != feature:
        raise ValueError('Wrong saved feature branch')
    m = FeatureCritic(feature).eval().requires_grad_(False)
    m.load_state_dict(ck['model'])
    return m, {'path':str(path),'fit_steps':fit['test'][name]['fit_steps'],
               'selected_step':ck['selected_step'],'feature':feature}


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan['kind'] != 'signal_attribution' or not 1 <= plan['round'] <= 20:
        raise ValueError('Requires preregistered toy attribution round')
    if plan['steps'] != [0,25,300,1000] or plan['features'] != ['plain','both']:
        raise ValueError('This matched historical attribution uses four fixed snapshots/two judges')
    output = toy_path(output); campaign = toy_path(plan['campaign'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    start = time.monotonic()
    atomic_json(output/'process.json',{'pid':os.getpid(),'start_time':time.time()})
    report = {'state':'running','round':plan['round'],'points':{},'scope':plan['scope']}
    arrays, bce_series, structure_series = {}, {}, {}
    def emit(row):
        atomic_json(output/'status.json',{'state':'running','active':row,'elapsed_seconds':time.monotonic()-start})
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        cfg, source = load_source(); data = Data(cfg)
        common_positive = None
        for step in plan['steps']:
            emit({'phase':'generating_counterfactuals','policy_step':step})
            model = source
            if step:
                ck = torch.load(HISTORY/f'round_{step}/full/state.pt',map_location='cpu',weights_only=True)
                if ck['step'] != step:raise ValueError('Policy step mismatch')
                model = film.NonlinearFiLM(cfg,source.state_dict(),width=32).eval()
                model.load_state_dict(ck['model'])
            panels,health,pool = swaps(model,data,plan['grid'],plan['samples_per_condition'],
                plan['donors_per_condition'],plan['sample_seed'],return_pool=True)
            if common_positive is None:common_positive = panels['A']['positive']
            if not all(torch.equal(p['positive'],common_positive) for p in panels.values()):
                raise ValueError('Truth pairing changed across counterfactuals/policies')
            g = plan['grid']; k = plan['truth_shape_per_cell']
            truth_ids = torch.arange(8)[None,:,None].expand(g,-1,k)
            ty = truth_shape(truth_ids,data.centers,native.generator(plan['sample_seed']+300000))
            physical = structure(pool,data.centers)
            q = probabilities(pool['ids'])
            point = {'cells':health,'structure':{n:float(v.mean()) for n,v in physical.items()},'judges':{}}
            arrays[str(step)] = {'c':pool['c'],'q':q,'p':pool['p'],'structure':physical,'judges':{}}
            for feature in plan['features']:
                judge,provenance = load_judge(campaign,step,feature)
                scores = {n:predictions(judge,p) for n,p in panels.items()}
                if not all(torch.isfinite(v).all() for ss in scores.values() for v in ss.values()):
                    raise FloatingPointError('Nonfinite judge prediction')
                bces = {n:bce_by_condition(s,g) for n,s in scores.items()}
                for swap in ('B','C','D'):
                    bce_series[f'step{step}/{feature}/{swap}_minus_A'] = bces[swap]-bces['A']
                for swap in ('B','C'):
                    bce_series[f'step{step}/{feature}/{swap}_minus_D'] = bces[swap]-bces['D']
                tmean,gmean = cell_means(judge,pool,ty)
                decomposition = score_decomposition(pool['p'],q,tmean,gmean,data.centers)
                point['judges'][feature] = {'provenance':provenance,
                    'swaps':{n:score_metrics(s) for n,s in scores.items()},
                    'logit_gap':{n:float(v.mean()) for n,v in decomposition.items()}}
                arrays[str(step)]['judges'][feature] = {'scores':scores,'bce_by_condition':bces,
                    'decomposition':decomposition,'truth_cell_means':tmean,'generator_cell_means':gmean}
                emit({'phase':'judge_scored','policy_step':step,'feature':feature})
            # Difference of differences cancels each judge's truth/truth control.
            for swap in ('B','C','D'):
                bce_series[f'step{step}/Fourier_minus_plain/{swap}_minus_A'] = (
                    bce_series[f'step{step}/both/{swap}_minus_A']-bce_series[f'step{step}/plain/{swap}_minus_A'])
            if step:
                for n in ('order1_mode_l2_debiased','order2_mode_l2_debiased','order3_mode_l2_debiased'):
                    structure_series[f'step{step}_minus_0/{n}'] = physical[n]-arrays['0']['structure'][n]
            report['points'][str(step)] = point
            atomic_checkpoint(output/f'step{step}_pool.pt',pool)
            atomic_checkpoint(output/'attribution_arrays.pt',arrays)
            atomic_json(output/'report.json',report)
        report['bce_contrasts'] = simultaneous_mean_intervals(bce_series,plan['bootstrap_repeats'],710097)
        report['structure_contrasts'] = simultaneous_mean_intervals(structure_series,plan['bootstrap_repeats'],710098)
        report.update(state='completed',elapsed_seconds=time.monotonic()-start)
        atomic_json(output/'report.json',report)
        lines = ['# Frozen plain/Fourier signal attribution','','A: truth modes + truth shape; B: generator modes + truth shape;',
                 'C: truth modes + generator shape; D: unchanged generator. Positive class is always independent truth.',
                 '', 'These are saved fresh judges, NOT classifiers refit on swaps. BCE/AUC changes are functional diagnostics.',
                 '', '| Policy step | Judge | D AUC | A BCE | B BCE | C BCE | D BCE |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for step,point in report['points'].items():
            for f,j in point['judges'].items():
                ss=j['swaps']; lines.append(f"| {step} | {f} | {ss['D']['auc']:.5f} | "+
                    ' | '.join(f"{ss[s]['bce']:.5f}" for s in ('A','B','C','D'))+' |')
        lines += ['', 'Detailed order-wise signed score decomposition, actual distribution errors, paired contrasts,',
                  'and provenance are in report.json. Descriptive intervals resample grid condition nodes,',
                  'not individual samples; they exclude training-seed and grid-quadrature uncertainty.',
                  'No policy update or target modification occurred. Review before the next scientific round.','']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json',{'state':'awaiting_review','round':plan['round'],
                    'elapsed_seconds':report['elapsed_seconds']})
        return report
    except Exception as exc:
        atomic_json(output/'status.json',{'state':'failed','error':repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.plan,args.output)
