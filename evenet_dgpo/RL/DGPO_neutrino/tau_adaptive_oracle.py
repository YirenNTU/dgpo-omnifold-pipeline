"""Round-held oracle strength from paired held-out classifier discrimination.

AUC uses uncapped head logits and independent events, not candidate weight ESS.
The paired influence SE includes positive/negative covariance within each event.
It is an asymptotic calibration diagnostic, not a guarantee of distribution closure.
"""
import copy
import numpy as np

DEFAULT_GATE = dict(method='heldout_paired_auc_lcb', max_alpha=.3,
                    calibration_events=11903, calibration_seed=4203,
                    generation_seed=8642, noise_z=1.96)


def validate_gate(cfg):
    if cfg != DEFAULT_GATE:
        raise ValueError('Adaptive oracle requires the declared held-out AUC gate protocol')


def paired_auc(truth_logits, generated_logits):
    p,q = np.asarray(truth_logits,dtype=float),np.asarray(generated_logits,dtype=float)
    if p.ndim != 1 or p.shape != q.shape or len(p)<20 or not np.isfinite([p,q]).all():
        raise ValueError('Require finite paired event scores, n>=20')
    sp,sq = np.sort(p),np.sort(q);n=len(p)
    vp=(np.searchsorted(sq,p,side='left')+np.searchsorted(sq,p,side='right'))/(2*n)
    vq=1-(np.searchsorted(sp,q,side='left')+np.searchsorted(sp,q,side='right'))/(2*n)
    auc=float(vp.mean())
    se=float(np.std(vp+vq,ddof=1)/np.sqrt(n))
    return dict(auc=auc,auc_se=se,events=n)


def next_state(cfg, previous, stats, *, round_id, step):
    validate_gate(cfg)
    auc,se=stats['auc'],stats['auc_se']
    if not (0<=auc<=1 and np.isfinite(se) and se>=0):raise ValueError('Invalid AUC/SE')
    if auc+cfg['noise_z']*se < .5:
        raise ValueError('Significantly inverted classifier; do not mistake inversion for closure')
    gap=max(auc-.5-cfg['noise_z']*se,0.)
    if previous is None:
        if step!=0 or round_id!=0:raise ValueError('Initial adaptive calibration must be epoch287 round0')
        if gap<=0:raise ValueError('Initial head has no resolved discrimination; baseline undefined')
        baseline=gap
    else:
        if previous['config']!=cfg or round_id!=previous['round_id']+1 or step<previous['policy_step']:
            raise ValueError('Adaptive oracle state/config/round mismatch')
        baseline=previous['baseline_gap']
    alpha=float(cfg['max_alpha']*np.clip(gap/baseline,0,1))
    return dict(version=1,config=copy.deepcopy(cfg),baseline_gap=baseline,alpha=alpha,
                round_id=int(round_id),policy_step=int(step),resolved_gap=gap,**stats)


def live_alpha(oracle_cfg, reward):
    cfg=oracle_cfg.get('adaptive')
    state=getattr(reward,'adaptive_oracle_state',None)
    if cfg is None:
        if state is not None:raise ValueError('Cannot resume adaptive reward as fixed oracle')
        return float(oracle_cfg['alpha'])
    validate_gate(cfg)
    if (state is None or state.get('version')!=1 or state['config']!=cfg
            or state['round_id']!=reward.round_id or state['policy_step']!=reward.denominator_step
            or not 0<=state['alpha']<=cfg['max_alpha']):
        raise ValueError('Missing/stale adaptive oracle calibration; no silent fixed-alpha fallback')
    if float(oracle_cfg['alpha'])!=cfg['max_alpha']:raise ValueError('Oracle cap mismatch')
    return float(state['alpha'])
