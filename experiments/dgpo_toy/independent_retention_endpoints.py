"""Independent-condition frozen-judge check, no fitting or policy updates.

The legacy evaluation seed970093 equals the first reward-fit training seed.
Visible rotations/noise differ, but latent scalar conditions overlap. Keep the
archived diagnostic curves; this new panel checks ALL endpoints without changing
the experiment, checkpoints or success thresholds. Fresh audits were already
independent and are unaffected.
"""
import argparse
import json
from pathlib import Path
import torch
from . import closed_loop_lab as lab
from . import relational_retention as retention
from .classifier_signal_attribution import simultaneous_mean_intervals
from .nonperiodic_cube import Data,panel
from .relational_conditioning import RelationalData
from .truth_pretrain import atomic_json,atomic_checkpoint


@torch.no_grad()
def run(root):
    root=lab.toy_path(root);out=root/'independent_endpoint_check'
    out.mkdir(exist_ok=False)
    p2=json.loads((root/'round02/plan.json').read_text())
    p3=json.loads((root/'round03/plan.json').read_text())
    plan={'seed':930207,'contexts':16384,'draws_per_context':1,
          'policy_updates':0,'fits':0,
          'reason':'Legacy frozen-reward panel reused latent condition RNG prefix of reward '
                   'fit training pool. Independent confirmation929927 and final cold audits '
                   'use different seeds already. Recheck every fixed300 endpoint together.',
          'scope':'Independent IID conditions and noise; same fixed original residual judge. '
                  'No new training seed and no endpoint/threshold selection.'}
    atomic_json(out/'plan.json',plan)
    cfg,source=lab.load_source();data=RelationalData(Data(cfg))
    judge,_=retention.installed_reward(p2)
    specs=[('baseline',p2,'raw',None),
           ('raw_fixed',p2,'raw',root/'round02/raw_step300.pt'),
           ('raw_refit',p2,'raw',root/'round05/periodic_step300.pt'),
           ('reference_only',p2,'raw',root/'round06/periodic_step300.pt'),
           ('relative_fixed',p3,'relative',root/'round03/relative_step300.pt'),
           ('relative_refit',p3,'relative',root/'round07/periodic_step300.pt')]
    values={};anchor=None
    for name,old,arm,path in specs:
        model=retention.make_models(source,cfg,old)[arm]
        if path:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(ck['model'],strict=True)
        pp=panel(data,plan['contexts'],plan['seed'],model)
        if anchor is None:
            anchor={k:pp[k] for k in ('c','positive')}
        for key in anchor:
            if not torch.equal(anchor[key],pp[key]):raise ValueError('Lost paired condition/truth')
        values[name]=lab.predictions(judge,pp)['negative'].double()
        print(name,float((values[name]-values['baseline']).mean()),flush=True)
    columns={f'{name}_gain':rr-values['baseline'] for name,rr in values.items() if name!='baseline'}
    columns.update(raw_refit_minus_fixed=values['raw_refit']-values['raw_fixed'],
                   raw_refit_minus_reference_only=values['raw_refit']-values['reference_only'],
                   relative_refit_minus_fixed=values['relative_refit']-values['relative_fixed'],
                   relative_refit_minus_raw_refit=values['relative_refit']-values['raw_refit'])
    report={'state':'completed','plan':plan,
            'contrasts':simultaneous_mean_intervals(columns,2000,930208),
            'scope':'All nine contrasts simultaneous; conditional on the trained models, '
                    'not across-training-seed robustness or proof of distribution closure.'}
    atomic_checkpoint(out/'arrays.pt',{'scores':values,'condition':anchor['c']})
    atomic_json(out/'report.json',report)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args();torch.set_num_threads(1);run(args.root)
