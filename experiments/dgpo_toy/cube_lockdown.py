"""Function-matched k8 representation x reference-penalty experiment."""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from .conditional import Config, Denoiser, alpha_sigma, generator, ddim, policy_train
from .parity_cube import CubeDistribution, ModeReward, cube_metrics
from .cube_frequency import summarize
from .coverage_budget import flatten_metrics
from .truth_pretrain import atomic_json, atomic_checkpoint

BASES=("raw", "polynomial", "fourier")
ARMS=tuple((basis,coef) for basis in BASES for coef in (0.,1.))


def condition_features(c,basis):
    """Eight fixed features, E[f]=0 and E[f_i^2]=1 under uniform c."""
    if basis=="raw":return c.expand(*c.shape[:-1],8)*0
    if basis=="fourier":
        phase=c*c.new_tensor([1.,2.,4.,8.])*math.pi
        return math.sqrt(2)*torch.cat((phase.sin(),phase.cos()),-1)
    if basis!="polynomial":raise ValueError(basis)
    p0=torch.ones_like(c);p1=c;parts=[math.sqrt(3)*p1]
    for degree in range(2,9):
        p2=((2*degree-1)*c*p1-(degree-1)*p0)/degree
        parts.append(math.sqrt(2*degree+1)*p2);p0,p1=p1,p2
    return torch.cat(parts,-1)


class ConditionDenoiser(Denoiser):
    """Zero-initialized input adapter preserves the entire initial denoiser."""
    def __init__(self,cfg,basis,state):
        super().__init__(cfg);self.load_state_dict(state);self.basis=basis
        self.condition_adapter=torch.nn.Linear(8,cfg.hidden,bias=False)
        torch.nn.init.zeros_(self.condition_adapter.weight)

    def predict_features(self,x,t,c):
        t=t.expand(x.shape[:-1]);c=c.expand(*x.shape[:-1],c.shape[-1])
        tf=torch.stack([t,(math.pi*t).sin(),(math.pi*t).cos(),
            (2*math.pi*t).sin(),(2*math.pi*t).cos()],-1)
        hidden=self.network[0](torch.cat([x,c,tf],-1))
        hidden=hidden+self.condition_adapter(condition_features(c,self.basis))
        hidden=self.network[1:-1](hidden)
        return alpha_sigma(t)[1][...,None]*self.network[-1](hidden),hidden


def contrast(after,before):
    d=(after-before).detach().double().mean(-1)
    se=float(d.std(unbiased=True)/math.sqrt(len(d)));gain=float(d.mean())
    # Conservative per-comparison intervals; separately require all rollout seeds.
    return {"gain":gain,"se":se,"lo":gain-3.5*se,"hi":gain+3.5*se,"z":3.5}


@torch.no_grad()
def evaluate(model,data,cfg,seed):
    n=cfg.eval_events;c=((torch.arange(n,dtype=torch.float32)+.5)/n*2-1)[:,None]
    z=torch.randn(n,cfg.candidates,3,generator=generator(seed))
    y=torch.cat([ddim(model,cc[:,None],zz,cfg.ddim_steps) for cc,zz in zip(c.split(128),z.split(128))])
    ctx=c[:,None].expand(-1,cfg.candidates,-1)
    r=ModeReward()(y,ctx,data);h=data.joint_signal(y,ctx)
    parity=torch.where(y>=0,1.,-1.).prod(-1)
    moment=data.condition_signal(ctx[...,0])*parity
    stats=cube_metrics(y.flatten(0,1),ctx.flatten(0,1),data)
    stats.update(reward_mean=float(r.mean()),group_hit_fraction=float(h.amax(1).mean()))
    return {"reward":r,"hits":h,"moment":moment},summarize(stats,r)


@torch.no_grad()
def verify_initial(source,models,cfg):
    rng=generator(800017);c=2*torch.rand(256,1,generator=rng)-1
    z=torch.randn(256,8,3,generator=rng);t=torch.rand(256,8,generator=rng)
    v=source(z,t,c[:,None]);y=ddim(source,c[:,None],z,cfg.ddim_steps)
    result={}
    for name,m in models.items():
        vv=m(z,t,c[:,None]);yy=ddim(m,c[:,None],z,cfg.ddim_steps)
        result[name]={"velocity_exact":torch.equal(v,vv),"samples_exact":torch.equal(y,yy),
            "parameter_count":sum(p.numel() for p in m.parameters())}
        if not result[name]["velocity_exact"] or not result[name]["samples_exact"]:
            raise RuntimeError("Function matching failed; stop before training")
    if len({v['parameter_count'] for v in result.values()})!=1:
        raise RuntimeError("Adapter parameter counts must match")
    return result


def conclusions(report):
    seeds=report['seeds'];arms=report['arms']
    if len(arms)!=len(seeds)*6:return {"state":"pending_all_prespecified_arms"}
    def result(seed,basis,coef,step):return arms[f'{seed}_{basis}_v{coef}']['evaluations'][str(step)]
    final=report['steps'];check=min(300,final);out={}
    for coef in (0,1):
        out[f'raw_v{coef}_material_gain_all_seeds']=all(result(s,'raw',coef,final)['gains']['reward']['lo']>.05 for s in seeds)
        out[f'raw_v{coef}_small_gain_all_seeds']=all(-.05<result(s,'raw',coef,final)['gains']['reward']['lo'] and result(s,'raw',coef,final)['gains']['reward']['hi']<.05 for s in seeds)
    out['delay_supported']=final>check and out['raw_v1_material_gain_all_seeds'] and all(result(s,'raw',1,check)['gains']['reward']['hi']<.05 for s in seeds)
    for name in ('fourier_minus_raw_v1','fourier_minus_polynomial_v1','polynomial_minus_raw_v1','raw_v0_minus_v1'):
        out[name+'_material_all_seeds']=all(report['contrasts'][str(s)][str(final)][name]['reward']['lo']>.05 for s in seeds)
    out['frequency_basis_rescue_supported']=out['fourier_minus_raw_v1_material_all_seeds'] and out['fourier_minus_polynomial_v1_material_all_seeds'] and all(result(s,'fourier',1,final)['gains']['reward']['lo']>.05 for s in seeds)
    out['penalty_sufficient_for_raw_plateau_at_budget']=out['raw_v0_material_gain_all_seeds'] and out['raw_v1_small_gain_all_seeds'] and out['raw_v0_minus_v1_material_all_seeds']
    out['scope']='Conditional on ONE pretrained k8 checkpoint; no production claim, no proof of unique or permanent cause.'
    out['state']='completed_prespecified_comparisons'
    return out


def run(source,output,steps=3000,seeds=(17,23,41),wandb_mode='offline'):
    if steps<300 or not seeds or len(set(seeds))!=len(seeds):raise ValueError('Need >=300 updates and distinct seeds')
    saved=torch.load(source/'best_pretrain.pt',map_location='cpu',weights_only=True)
    if saved.get('condition_frequency')!=8:raise ValueError('Use fixed k8 pretrain checkpoint')
    cfg=replace(Config(**saved['config']),policy_steps=steps,eval_every=300,eval_events=512)
    evalcfg=replace(cfg,eval_events=8192)
    data=CubeDistribution(cfg,continuous=True,reference_sharpness=4.,condition_frequency=8)
    original=Denoiser(cfg);original.load_state_dict(saved['model']);original.eval()
    models={b:ConditionDenoiser(cfg,b,saved['model']).eval() for b in BASES}
    matching=verify_initial(original,models,cfg)
    output.mkdir(parents=True,exist_ok=False)
    report={'state':'running','source':str(source.resolve()),'source_step':saved['step'],
        'steps':steps,'seeds':list(seeds),'config':asdict(cfg),'initial_matching':matching,
        'arms':{},'contrasts':{},'predeclared_reward_margin':.05,'evaluation_z':3.5,
        'scope':'All arms function-matched at step0; polynomial and Fourier eight unit-variance features and identical added parameters. No pretraining/classifier/reward modifications.',
        'endpoint_seed':880017,'monitor_seed':870017,'policy_native_monitor_seed':860017}
    import wandb
    wb=wandb.init(project='dgpo-toy',mode=wandb_mode,dir=str(output.resolve()),
        name='What blocks conditional reward? | matched start | basis x penalty | three seeds',
        group='Conditional cube mechanism',config=report,tags=['k8','function-matched','raw-no-ema','factorial'])
    report['wandb']={'id':wb.id,'directory':wb.dir,'mode':wandb_mode};start=time.monotonic()
    atomic_json(output/'report.json',report)
    try:
        baseline,stats=evaluate(original,data,evalcfg,880017);report['initial']=stats
        atomic_checkpoint(output/'initial_eval.pt',baseline)
        arrays={}
        with (output/'progress.jsonl').open('w') as log:
            def emit(row):
                line=json.dumps(row,allow_nan=False);log.write(line+'\n');log.flush();print(line,flush=True)
                wb.log(flatten_metrics(row,row['arm_key']+'/'+row['phase']+'/'))
            for seed in seeds:
                arrays[str(seed)]={}
                for basis,coef in ARMS:
                    key=f'{seed}_{basis}_v{int(coef)}';directory=output/key;directory.mkdir()
                    initial=models[basis];before=copy.deepcopy(initial.state_dict())
                    report.update(active_arm=key,active_step=0)
                    arm={'seed':seed,'basis':basis,'velocity_coefficient':coef,'evaluations':{},'state':'running'}
                    report['arms'][key]=arm;arrays[str(seed)][key]={}
                    atomic_json(output/'report.json',report)
                    def save(step,m,opt,rng,history):
                        report['active_step']=step
                        if step in (300,steps):
                            panel,physical=evaluate(m,data,evalcfg,880017)
                            gains={name:contrast(panel[name],baseline[name]) for name in panel}
                            arm['evaluations'][str(step)]={'physical':physical,'gains':gains}
                            arrays[str(seed)][key][str(step)]=panel
                            atomic_checkpoint(directory/f'evaluation_{step}.pt',panel)
                            emit({'arm_key':key,'phase':'endpoint','step':step,'physical':physical,'gains':gains})
                        if step==1 or step%300==0 or step==steps:
                            atomic_checkpoint(directory/'state.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),
                                'rng':rng.get_state(),'history':history,'step':step,'config':asdict(cfg),
                                'velocity_coefficient':coef,'basis':basis,'seed':seed,'monitor_seed':860017,
                                'condition_frequency':8,'source':report['source']})
                        if step==1 or step%25==0 or step==steps:
                            report['elapsed_seconds']=time.monotonic()-start;atomic_json(output/'report.json',report)
                    _,history=policy_train('dgpo',initial,ModeReward(),data,cfg,seed,860017,
                        lambda row:emit({**row,'arm_key':key}),save,velocity_coefficient=coef)
                    arm.update(state='completed',completed_steps=len(history),source_unchanged=all(torch.equal(v,before[k]) for k,v in initial.state_dict().items()))
                    atomic_json(output/'report.json',report)
                pairs={'fourier_minus_raw_v1':('fourier_v1','raw_v1'),
                    'fourier_minus_polynomial_v1':('fourier_v1','polynomial_v1'),
                    'polynomial_minus_raw_v1':('polynomial_v1','raw_v1'),
                    'raw_v0_minus_v1':('raw_v0','raw_v1')}
                report['contrasts'][str(seed)]={}
                for step in set((300,steps)):
                    report['contrasts'][str(seed)][str(step)]={}
                    for label,(a,b) in pairs.items():
                        aa=arrays[str(seed)][f'{seed}_{a}'][str(step)];bb=arrays[str(seed)][f'{seed}_{b}'][str(step)]
                        report['contrasts'][str(seed)][str(step)][label]={n:contrast(aa[n],bb[n]) for n in aa}
                atomic_json(output/'report.json',report)
                del arrays[str(seed)]
        report['decision']=conclusions(report);report['state']='completed'
    except BaseException as exc:
        report.update(state='failed_or_interrupted',error=repr(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        atomic_json(output/'report.json',report);wb.summary.update(flatten_metrics(report))
        wb.finish(exit_code=0 if report['state']=='completed' else 1)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/dgpo_toy/cube_frequency_v1/k8/pretrain'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,23,41])
    p.add_argument('--wandb-mode',choices=('offline','online','disabled'),default='offline')
    args=p.parse_args();torch.set_num_threads(1);run(**vars(args))
