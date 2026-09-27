"""Cross-score saved raw/Fourier conditioning arms; no fitting or policy updates."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from . import classifier_signal_attribution as a
from . import conditional as native
from .closed_loop_lab import ConditioningAblation, FeatureCritic, load_source, predictions, score_metrics, toy_path
from .cube_swap import swaps, truth_shape
from .nonperiodic_cube import Data
from .truth_pretrain import atomic_checkpoint, atomic_json


def load_inputs(source):
    source=toy_path(source)
    report=json.loads((source/'report.json').read_text())
    plan=json.loads((source/'plan.json').read_text())
    if report['state']!='completed' or not report['valid']:
        raise ValueError('Source conditioning round did not complete valid audits')
    cfg,base=load_source()
    policies={'baseline':base}
    provenance={}
    for arm in ('raw','fourier'):
        path=source/f'{arm}_step1000.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck['step']!=1000 or ck['conditioning']!=plan['conditioning'][arm]:
            raise ValueError('Policy step or conditioning mismatch')
        m=ConditioningAblation(cfg,base.state_dict(),**ck['conditioning']).eval()
        m.load_state_dict(ck['model']);policies[arm]=m
        provenance[arm]={'policy':str(path),'conditioning':ck['conditioning']}
    judges={}
    for arm in ('baseline','raw','fourier'):
        folder=source/('audit0' if arm=='baseline' else 'audit1000')
        name=f'{arm}__both'
        fit=json.loads((folder/'fit_report.json').read_text())['test'][name]
        if not fit['valid'] or fit['fit_steps']<8000:
            raise ValueError('Undertrained source audit')
        path=folder/f'{name}_best.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        if ck['feature']!='both':raise ValueError('Use saved full Fourier judges')
        m=FeatureCritic('both').eval().requires_grad_(False)
        m.load_state_dict(ck['model']);judges[arm]=m
        provenance.setdefault(arm,{}).update(judge=str(path),fit=fit)
    return cfg,policies,judges,provenance


def compare_arrays(arrays,repeats):
    """Compare the two policies using each SAME fixed judge, plus own-judge pairs."""
    bce={}
    for judge in ('baseline','raw','fourier'):
        for swap in ('B','C','D'):
            bce[f'judge_{judge}/{swap}/fourier_minus_raw']= (
                arrays['fourier']['judges'][judge]['bce'][swap]-arrays['raw']['judges'][judge]['bce'][swap])
    for swap in ('B','C','D'):
        f,r=arrays['fourier']['judges']['fourier']['bce'],arrays['raw']['judges']['raw']['bce']
        bce[f'own_judges/{swap}_minus_A/fourier_minus_raw']=(f[swap]-f['A'])-(r[swap]-r['A'])
    structure={}
    for k in (1,2,3):
        key=f'order{k}_mode_l2_debiased'
        structure[f'{key}/fourier_minus_raw']=arrays['fourier']['structure'][key]-arrays['raw']['structure'][key]
        for arm in ('raw','fourier'):
            structure[f'{key}/{arm}_minus_baseline']=arrays[arm]['structure'][key]-arrays['baseline']['structure'][key]
    return {'bce':a.simultaneous_mean_intervals(bce,repeats,710097),
            'structure':a.simultaneous_mean_intervals(structure,repeats,710098)}


def run(plan_path,output):
    plan=json.loads(plan_path.read_text())
    if plan['kind']!='conditioning_attribution' or not 1<=plan['round']<=20:
        raise ValueError('Invalid toy attribution plan')
    cfg,policies,judges,provenance=load_inputs(plan['source_round'])
    output=toy_path(output);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'process.json',{'pid':os.getpid(),'start_time':time.time()})
    start=time.monotonic()
    report={'state':'running','round':plan['round'],'provenance':provenance,'points':{},'scope':plan['scope']}
    arrays={};data=Data(cfg);anchor=None
    def emit(row):
        atomic_json(output/'status.json',{'state':'running','active':row,'elapsed_seconds':time.monotonic()-start})
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        for arm,policy in policies.items():
            emit({'phase':'generating','policy':arm})
            panels,health,pool=swaps(policy,data,plan['grid'],plan['samples_per_condition'],
                plan['donors_per_condition'],plan['sample_seed'],return_pool=True)
            if anchor is None:anchor=panels['A']
            if not all(torch.equal(panels['A'][k],anchor[k]) for k in anchor):
                raise ValueError('Common truth panel lost')
            if not all(torch.equal(p['positive'],anchor['positive']) for p in panels.values()):
                raise ValueError('Common positive population lost')
            g,k=plan['grid'],plan['truth_shape_per_cell']
            truth_ids=torch.arange(8)[None,:,None].expand(g,-1,k)
            truth_y=truth_shape(truth_ids,data.centers,native.generator(plan['sample_seed']+300000))
            physical=a.structure(pool,data.centers);q=a.probabilities(pool['ids'])
            arrays[arm]={'structure':physical,'q':q,'p':pool['p'],'c':pool['c'],'judges':{}}
            point={'cells':health,'structure':{n:float(v.mean()) for n,v in physical.items()},'judges':{}}
            for name,judge in judges.items():
                scores={n:predictions(judge,p) for n,p in panels.items()}
                if not all(torch.isfinite(v).all() for s in scores.values() for v in s.values()):
                    raise FloatingPointError('Nonfinite saved judge predictions')
                tmean,gmean=a.cell_means(judge,pool,truth_y)
                decomposition=a.score_decomposition(pool['p'],q,tmean,gmean,data.centers)
                point['judges'][name]={'swaps':{n:score_metrics(s) for n,s in scores.items()},
                    'logit_gap':{n:float(v.mean()) for n,v in decomposition.items()}}
                arrays[arm]['judges'][name]={'scores':scores,'bce':{n:a.bce_by_condition(s,g) for n,s in scores.items()},
                    'decomposition':decomposition,'truth_cell_means':tmean,'generator_cell_means':gmean}
                emit({'phase':'scored','policy':arm,'judge':name})
            report['points'][arm]=point
            atomic_checkpoint(output/f'{arm}_pool.pt',pool)
            atomic_checkpoint(output/'attribution_arrays.pt',arrays)
            atomic_json(output/'report.json',report)
        report['contrasts']=compare_arrays(arrays,plan['bootstrap_repeats'])
        report.update(state='completed',elapsed_seconds=time.monotonic()-start)
        atomic_json(output/'report.json',report)
        lines=['# Round6: paired conditioning mode/shape attribution','',
               'Same saved judges cross-score all policies. A truth/truth; B generator modes/truth shape;',
               'C truth modes/generator shape; D unmodified generator. No fitting or training seed replication.','',
               '| Policy | Judge | A AUC | B AUC | C AUC | D AUC | D BCE |',
               '|---|---|---:|---:|---:|---:|---:|']
        for arm,p in report['points'].items():
            for name,j in p['judges'].items():
                ss=j['swaps'];lines.append(f'| {arm} | {name} | '+
                    ' | '.join(f"{ss[s]['auc']:.6f}" for s in 'ABCD')+f" | {ss['D']['bce']:.6f} |")
        lines+=['','Paired conditional-grid bootstrap contrasts and exact order decompositions: report.json.',
                'These are fixed-judge counterfactual sensitivities, NOT new best-response fits on swaps.',
                'Descriptive uncertainty excludes training seeds, grid quadrature and donor-pool uncertainty.','']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json',{'state':'awaiting_review','round':plan['round'],'elapsed_seconds':report['elapsed_seconds']})
        return report
    except Exception as exc:
        atomic_json(output/'status.json',{'state':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);run(args.plan,args.output)
