"""User-launched 16-GPU frozen-classifier sampling convergence, no training."""
from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.tau_sampling_convergence import convergence_report, group_report


def read_settings(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['workers']!=16 or cfg['batch_size']!=1024 or cfg['prefixes'] not in ([1,2,4,8],[1,2,4,8,16,32],[1,2,4,8,16,32,64])
        or cfg['expected_events']!=119002 or cfg['feature_batch_size']<1 or cfg['bootstrap']<20):
        raise ValueError('Requires full validation, 16 GPUs, batch1024, nested K through 8, 32 or 64')
    if max(cfg['prefixes'])>8 and not (cfg.get('extend_from') and cfg.get('extend_run')):
        raise ValueError('K32/K64 must reuse the completed K8 experiment')
    if len(cfg['logger']['name'])>96 or len(cfg['logger']['name'].split(' | '))<3:
        raise ValueError('Invalid W&B display name')
    return cfg


def validate_extension(cfg,inputs):
    """Read-only check before copying old shards. No silent regeneration on mismatch."""
    if not cfg.get('extend_from'):
        return dict(cfg,inherited_candidates=0)
    source=Path(cfg['extend_from']).resolve()
    if not (source/'COMPLETE').is_file(): raise ValueError('Extension source is incomplete')
    old=json.loads((source/'manifest.json').read_text())
    if json.loads((source/'wandb.json').read_text())['id']!=cfg['extend_run']:
        raise ValueError('Extension source run differs')
    keys=('classifier_run','classifier_checkpoint','generator_checkpoint','events','conditions',
          'runtime','workers','batch_size','feature_batch_size','seed','ddim_steps','packing_spec',
          'weights','ratio_transform','condition_pt_edges','historical_cij_errors','bootstrap')
    for key in keys:
        if key not in old or key not in cfg or old[key]!=cfg[key]:
            raise ValueError('Extension protocol changed: '+key)
    if old['prefixes']!=[1,2,4,8] or cfg['prefixes'] not in ([1,2,4,8,16,32],[1,2,4,8,16,32,64]):
        raise ValueError('Only completed K8 to K32/K64 extension is supported')
    with np.load(source/'inputs.npz',allow_pickle=False) as f:
        if set(f.files)!=set(inputs): raise ValueError('Extension input schema differs')
        for key,value in inputs.items():
            if not np.array_equal(f[key],value): raise ValueError('Extension inputs changed: '+key)
    for rank in range(cfg['workers']):
        positions=np.arange(rank,cfg['conditions'],cfg['workers'])
        for j in range(8):
            with np.load(source/f'rank-{rank:02d}-candidate-{j:02d}.npz',allow_pickle=False) as f:
                if not np.array_equal(f['positions'],positions) or not np.array_equal(f['source_ids'],inputs['source_ids'][positions]):
                    raise ValueError('Extension shard identity mismatch')
                for key,shape in [('deltas',(2,2)),('logits',()),('tau',(15,)),('cij',(9,))]:
                    a=f[key]
                    if a.shape!=(len(positions),*shape) or not np.isfinite(a).all():
                        raise ValueError('Extension shard invalid: '+key)
    prior=json.loads((source/'convergence_report.json').read_text())
    if prior['events']!=cfg['conditions'] or prior['candidates']!=8:
        raise ValueError('Extension report population differs')
    return dict(cfg,extend_from=str(source),inherited_candidates=8)


def copy_inherited_shards(cfg):
    """Copy rather than mutate/symlink the historical source; preserve all byte values."""
    for rank in range(cfg['workers']):
        for j in range(cfg.get('inherited_candidates',0)):
            name=f'rank-{rank:02d}-candidate-{j:02d}.npz'
            shutil.copy2(Path(cfg['extend_from'])/name,Path(cfg['output'])/name)


def new_candidate_indices(cfg):
    return range(cfg.get('inherited_candidates',0),max(cfg['prefixes']))


def verify_inherited_report(cfg,report):
    if not cfg.get('inherited_candidates'): return
    prior=json.loads((Path(cfg['extend_from'])/'convergence_report.json').read_text())
    for k,row in prior['prefixes'].items():
        for key in ('unweighted','reweighted','matrix_ci95','error_change_ci95',
                    'candidate_ess','unweighted_error','reweighted_error'):
            if not np.allclose(row[key],report['prefixes'][k][key],atol=1e-10,rtol=1e-10):
                raise ValueError(f'Inherited K{k} endpoint changed: {key}')
    report['extension']=dict(source_run=cfg['extend_run'],source_directory=cfg['extend_from'],
        inherited_candidates=cfg['inherited_candidates'],old_prefixes_reproduced=True,
        new_candidates_per_condition=max(cfg['prefixes'])-cfg['inherited_candidates'])


def physics_features(delta, va, vb, kappas):
    from scripts.diagnose_ztautau_cij import tau_from_deltas, angles
    from scripts.train_conditional_spin_ratio import tau_features
    from scripts.tau_relative_inputs import relative_angles
    from scripts.diagnose_reweighted_cij import features
    a,b=tau_from_deltas(va,delta[:,0]),tau_from_deltas(vb,delta[:,1])
    sa,sb=angles(a,b,va,vb)
    return tau_features(a,b),relative_angles(a,b,va,vb),features(sa,sb,kappas)


def prepare(cfg):
    from scripts.sample_1110_cij import validation_pool
    from scripts.diagnose_ztautau_cij import source_ids, align, read_event_table, SOURCE_ID_COLUMNS
    from scripts.conditional_tau_preprocessing import apply_masked_feature
    source=Path(cfg['classifier_directory'])
    manifest=json.loads((source/'manifest.json').read_text())
    if not (source/'COMPLETE').is_file() or json.loads((source/'wandb.json').read_text())['id']!=cfg['classifier_run']:
        raise ValueError('Require completed pinned classifier run')
    if (manifest['head_kind']!='film' or manifest['ratio_objective']!='bce'
        or manifest['head_depth']!=3 or manifest['relative_dim']!=6):
        raise ValueError('Not the completed three-block relative-input FiLM classifier')
    if Path(manifest['test_events']).resolve()!=Path(cfg['events']).resolve():
        raise ValueError('Validation source differs from classifier evaluation')
    if json.loads((Path(cfg['events']).parent/'filter_manifest.json').read_text()).get('complete') is not True:
        raise ValueError('Require completed filtered validation dataset')
    backbone=manifest['backbone_manifest']
    if backbone['global_step']!=1110 or backbone['weights']!='raw_state_dict_only':
        raise ValueError('Require raw step1110 backbone')
    sampling=manifest['test_sample_manifest']
    if (Path(sampling['checkpoint']).resolve()!=Path(backbone['checkpoint']).resolve()
        or sampling['weights']!='raw_state_dict_only' or sampling['ddim_steps']!=20):
        raise ValueError('Generator must be the same raw1110 checkpoint as the frozen feature trunk')
    checkpoint=source/'best.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if (saved['head_kind']!='film' or saved['head_depth']!=3 or saved['relative_dim']!=6
        or saved['condition_normalization']!='masked_feature' or saved['packing_spec']!=manifest['packing_spec']):
        raise ValueError('Classifier checkpoint schema differs')
    pool=validation_pool(cfg['events'],{'packing_spec':saved['packing_spec']})
    ids=source_ids(pool)
    if len(ids)!=cfg['expected_events'] or pool['invalid_target_rows']!=0:
        raise ValueError('Unexpected validation count or rejected events')
    with np.load(source/'test_scores.npz',allow_pickle=False) as scores:
        wanted=scores['source_ids'].copy()
        old_logits=scores['generated_logits'].copy()
        if not np.array_equal(old_logits,scores['log_ratio']): raise ValueError('Non-raw classifier scores')
    order=align(ids,wanted)
    if len(order)!=len(ids): raise ValueError('Validation subset is not allowed')
    raw=pool['condition'][order].numpy()
    truth=pool['truth'][order].numpy()
    keys=[k for k in SOURCE_ID_COLUMNS if k in pool]
    required=keys+['event_weight','event_category','analyzing_power_a','analyzing_power_b']
    required += [f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('E','px','py','pz')]
    table=read_event_table(cfg['events'],required)
    columns={k:np.asarray(table[k].to_numpy()) for k in required}
    ix=align(source_ids(columns,keys),wanted)
    columns={k:v[ix] for k,v in columns.items()}
    va,vb=[np.stack([columns[f'lead_{leg}_visible_{c}'] for c in ('E','px','py','pz')],1) for leg in ('a','b')]
    category=columns['event_category'].astype(int)
    if not np.isin(category//10,range(1,5)).all() or not np.isin(category%10,range(1,5)).all():
        raise ValueError('Invalid decay category')
    onehot=np.eye(16,dtype=np.float32)[(category//10-1)*4+category%10-1]
    condition=apply_masked_feature(np.concatenate((raw,onehot),1),saved['condition_mean'].numpy(),
                                  saved['condition_scale'].numpy(),saved['packing_spec'])
    kappas=np.stack([columns['analyzing_power_'+leg] for leg in ('a','b')],1)
    if not np.isfinite(kappas).all() or (kappas==0).any(): raise ValueError('Invalid analyzing powers')
    tau,_,cij=physics_features(truth,va,vb,kappas)
    with np.load(source/'prepared.npz',allow_pickle=False) as p:
        take=p['split']==2
        for key,actual in [('source_ids',wanted),('condition',condition),('tau_truth',tau),
                           ('event_weight',columns['event_weight']),('category',category),('kappas',kappas)]:
            expected=p[key][take]
            matches=np.array_equal(actual,expected) if key=='source_ids' else np.allclose(actual,expected,rtol=1e-6,atol=1e-6)
            if not matches: raise ValueError('Historical test reconstruction differs: '+key)
        from scripts.diagnose_reweighted_cij import features
        expected=features(p['truth_a'][take].astype(float),p['truth_b'][take].astype(float),kappas)
        if not np.allclose(cij,expected,atol=1e-6,rtol=1e-6): raise ValueError('Matched Cij map changed')
    with np.load(Path(manifest['test_source'])/'candidates.npz',allow_pickle=False) as old:
        oi=align(source_ids(old,keys),wanted)
        old_deltas=old['deltas'][oi,0]
        if not np.allclose(old['truth_deltas'][oi],truth,atol=1e-6,rtol=0):
            raise ValueError('Historical truth deltas differ')
    inputs=dict(raw_condition=raw,condition=condition,truth_deltas=truth,visible_a=va,visible_b=vb,
        kappas=kappas,category=category,weight=columns['event_weight'],truth_tau=tau,truth_cij=cij,
        visible_pt_sum=np.linalg.norm(va[:,1:3],axis=1)+np.linalg.norm(vb[:,1:3],axis=1),
        old_deltas=old_deltas,old_logits=old_logits,source_ids=wanted)
    if (inputs['weight']<0).any() or not np.isfinite(inputs['weight']).all() or inputs['weight'].sum()<=0:
        raise ValueError('Invalid base weights')
    historical=json.loads((source/'cij_comparison.json').read_text())
    historical_errors=[historical['arms'][arm]['error'] for arm in ('unweighted','candidate_ratio')]
    cfg=dict(cfg,classifier_checkpoint=str(checkpoint),generator_checkpoint=backbone['checkpoint'],
        runtime=backbone['runtime'],ddim_steps=sampling['ddim_steps'],packing_spec=saved['packing_spec'],
        condition_pt_edges=manifest['condition_pt_edges'],conditions=len(wanted),
        classifier_selected_epoch=saved['epoch']+1,weights='raw_state_dict_only',historical_cij_errors=historical_errors,
        policy_updates=0,classifier_fits=0,ratio_transform='raw exp(logit), globally normalized only',
        historical_K1='Separate old sample realization, NOT the fresh nested K1 prefix',
        evaluation_status='Exploratory previously inspected validation population; fixed trained classifier')
    return cfg,inputs


@torch.no_grad()
def score_draw(policy,head,saved,batch,delta,condition,va,vb,kappas,device,microbatch):
    from scripts.tau_backbone_alignment import candidate_hidden
    # Match extract_worker (default highest) even when DDIM uses its original TF32 setting.
    torch.set_float32_matmul_precision('highest')
    features=[]
    for start in range(0,len(delta),microbatch):
        stop=start+microbatch
        sub={k:v[start:stop] if torch.is_tensor(v) and v.ndim and len(v)==len(delta) else v for k,v in batch.items()}
        features.append(candidate_hidden(policy,sub,delta[start:stop]))
    tau,relative,cij=physics_features(delta.cpu().numpy(),va,vb,kappas)
    stats=saved['relative_preprocessing']
    relative=((relative-np.asarray(stats['mean']))/np.asarray(stats['scale'])).astype('float32')
    candidate=np.concatenate((np.concatenate(features),relative,tau),1).astype('float32')
    s=head(torch.as_tensor(condition,device=device),torch.as_tensor(candidate,device=device)).cpu().numpy()
    if not np.isfinite(s).all(): raise ValueError('Nonfinite fresh ratio')
    return s,tau,cij


@torch.no_grad()
def worker(cfg,rank):
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec,unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from scripts.diagnose_h4_spike_coverage import load_raw_state
    from scripts.train_conditional_spin_ratio import build_classifier
    torch.set_num_threads(1)
    device=torch.device('cuda:0')
    torch.set_float32_matmul_precision('highest')
    global_config.load_yaml(cfg['runtime'])
    policy=build_evenet_on_device(global_config,load_normalization_dict(global_config),device).eval()
    source=torch.load(cfg['generator_checkpoint'],map_location='cpu',weights_only=False)
    load_raw_state(policy,source,'step1110'); del source
    policy.requires_grad_(False)
    saved=torch.load(cfg['classifier_checkpoint'],map_location='cpu',weights_only=True)
    head=build_classifier(saved).to(device).eval()
    head.load_state_dict(saved['state_dict'],strict=True); head.requires_grad_(False)
    spec=EventPackingSpec.from_dict(saved['packing_spec'])
    with np.load(Path(cfg['output'])/'inputs.npz',allow_pickle=False) as f:
        pos=np.arange(rank,cfg['conditions'],cfg['workers'])
        a={key:f[key][pos] for key in f.files}
    def batch_at(start,stop):
        b=unpack_event_inputs(torch.as_tensor(a['raw_condition'][start:stop],device=device),spec)
        b.pop('classification',None)
        b['x_invisible']=torch.zeros(stop-start,2,2,device=device)
        b['x_invisible_mask']=torch.ones(stop-start,2,dtype=torch.bool,device=device)
        return b
    # Reconstruct original features and scores before accepting any new prediction.
    v=min(64,len(pos))
    s,_,_=score_draw(policy,head,saved,batch_at(0,v),torch.as_tensor(a['old_deltas'][:v],device=device),
        a['condition'][:v],a['visible_a'][:v],a['visible_b'][:v],a['kappas'][:v],device,cfg['feature_batch_size'])
    parity=float(np.max(np.abs(s-a['old_logits'][:v])))
    if not np.allclose(s,a['old_logits'][:v],atol=1e-3,rtol=1e-4):
        raise ValueError(f'Historical classifier replay failed on rank{rank}: max error {parity}')
    sampler=DDIMSampler(device=device)
    runtime=yaml.safe_load(Path(cfg['runtime']).read_text())
    generation_precision=str(runtime['dgpo'].get('float32_matmul_precision','medium'))
    for j in new_candidate_indices(cfg):
        # Separate deterministic streams per rank and candidate. No truth/score-dependent sampling.
        torch.manual_seed(cfg['seed']+1000003*j+rank)
        draws,logits,taus,cijs=[],[],[],[]
        for start in range(0,len(pos),cfg['batch_size']):
            stop=min(start+cfg['batch_size'],len(pos)); batch=batch_at(start,stop)
            torch.set_float32_matmul_precision(generation_precision)
            delta=generate_neutrino_candidates(policy,batch,sampler,K=1,num_ddim_steps=cfg['ddim_steps'],
                                               device=device,parallel_chains=1)[0]
            s,t,c=score_draw(policy,head,saved,batch,delta,a['condition'][start:stop],
                a['visible_a'][start:stop],a['visible_b'][start:stop],a['kappas'][start:stop],device,cfg['feature_batch_size'])
            draws.append(delta.cpu().numpy());logits.append(s);taus.append(t);cijs.append(c)
            print(f'[sampling rank={rank}] candidate={j+1}/{max(cfg["prefixes"])} events={stop}/{len(pos)}',flush=True)
        np.savez(Path(cfg['output'])/f'rank-{rank:02d}-candidate-{j:02d}.npz',positions=pos,
            source_ids=a['source_ids'],deltas=np.concatenate(draws),logits=np.concatenate(logits),
            tau=np.concatenate(taus),cij=np.concatenate(cijs),replay_max_error=parity)
        progress=Path(cfg['output'])/f'progress-{rank:02d}.json'
        tmp=progress.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(samples=(j+1)*len(pos),candidates=j+1,rank=rank)))
        tmp.replace(progress)
    return rank


def merge(cfg,ids):
    n,k=len(ids),max(cfg['prefixes'])
    result={key:np.empty((n,k,*shape),dtype=dtype) for key,shape,dtype in
            [('deltas',(2,2),np.float32),('logits',(),np.float64),('tau',(15,),np.float32),('cij',(9,),np.float64)]}
    seen=np.zeros((n,k),int); parity=[]
    for rank in range(cfg['workers']):
        for j in range(k):
            with np.load(Path(cfg['output'])/f'rank-{rank:02d}-candidate-{j:02d}.npz') as f:
                p=f['positions']
                if (p<0).any() or (p>=n).any() or len(np.unique(p))!=len(p) or not np.array_equal(ids[p],f['source_ids']):
                    raise ValueError('Invalid inference shard identities')
                seen[p,j]+=1
                for key in result:
                    x=f[key]
                    if x.shape!=result[key][p,j].shape or not np.isfinite(x).all(): raise ValueError('Invalid shard '+key)
                    result[key][p,j]=x
                parity.append(float(f['replay_max_error']))
    if not (seen==1).all(): raise ValueError('Missing or duplicated candidate rows')
    return result,max(parity)


def finish(cfg,run):
    output=Path(cfg['output'])
    with np.load(output/'inputs.npz') as f: a={key:f[key] for key in f.files}
    data,parity=merge(cfg,a['source_ids'])
    np.savez_compressed(output/'samples_and_scores.npz',source_ids=a['source_ids'],**data)
    run.summary.update(dict(phase='analysis',historical_score_replay_max_error=parity))
    report=convergence_report(a['truth_cij'],data['cij'],data['logits'],a['weight'],cfg['prefixes'],cfg['bootstrap'],cfg['seed'])
    verify_inherited_report(cfg,report)
    report['conditional_tau']=group_report(a['truth_tau'],data['tau'],data['logits'],a['weight'],
        a['category'],a['visible_pt_sum'],cfg['condition_pt_edges'],cfg['prefixes'])
    # Historical K1 is a separate realization, never substituted for fresh prefix1.
    _,_,old_cij=physics_features(a['old_deltas'],a['visible_a'],a['visible_b'],a['kappas'])
    from scripts.tau_sampling_convergence import aggregates
    num,den,_,_=aggregates(old_cij[:,None],a['old_logits'][:,None],a['weight'],1)
    old=num.sum(0)/den.sum(0)[:,None]
    old_error=np.linalg.norm(old-np.asarray(report['truth']),axis=1)
    if not np.allclose(old_error,cfg['historical_cij_errors'],atol=1e-6,rtol=1e-6):
        raise ValueError('Historical Cij endpoint replay differs from pinned classifier report')
    report['historical_K1']=dict(unweighted=old[0].tolist(),reweighted=old[1].tolist(),
        error=old_error.tolist(),source_run=cfg['classifier_run'])
    path=output/'convergence_report.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import wandb
    matrix_rows=[]
    for k in cfg['prefixes']:
        row=report['prefixes'][str(k)]
        metrics={'candidates_per_condition':k}
        metrics.update({f'convergence/{key}':v for key,v in row.items() if isinstance(v,(float,int))})
        metrics.update({f'convergence/error_change_{bound}95':value for bound,value in zip(('lo','hi'),row['error_change_ci95'])})
        groups=report['conditional_tau'][str(k)]
        cats=[r for r in groups if r['group'].startswith('category_') and '_pt_' not in r['group']]
        joint=[r for r in groups if '_pt_' in r['group']]
        metrics['convergence/category_mass_tv']=.5*sum(abs(r['weighted_mass']-r['base_mass']) for r in cats)
        metrics['convergence/group_tau_error_unweighted']=sum(r['base_mass']*r['unweighted_error'] for r in joint)
        metrics['convergence/group_tau_error_reweighted']=sum(r['base_mass']*r['reweighted_error'] for r in joint)
        metrics['convergence/group_log_mean_ratio_rms']=float(np.sqrt(sum(r['base_mass']*r['log_mean_ratio']**2 for r in joint)))
        for j,axis in enumerate(('kk','kr','kn','rk','rr','rn','nk','nr','nn')):
            for arm in ('unweighted','reweighted'):
                metrics[f'Cij/{arm}/{axis}']=row[arm][j]
                matrix_rows.append([k,arm,axis,row[arm][j],report['truth'][j]])
            metrics[f'Cij/truth/{axis}']=report['truth'][j]
            if row['conditional_mc_se'] is not None:
                metrics[f'Cij/conditional_mc_se_reweighted/{axis}']=row['conditional_mc_se'][1][j]
        run.log(metrics)
        run.summary.update({f'K{k}/{key}':v for key,v in row.items() if isinstance(v,(float,int))})
        run.summary[f'K{k}/error_change_ci95']=row['error_change_ci95']
    run.log({'Cij/matrices':wandb.Table(columns=['K','arm','component','Cij','truth'],data=matrix_rows)})
    panels=[[int(k),r['start'],r['error_change'],*r['reweighted']] for k,rows in report['disjoint_panels'].items() for r in rows]
    run.log({'sampling/disjoint_panels':wandb.Table(columns=['K','start','error_change','kk','kr','kn','rk','rr','rn','nk','nr','nn'],data=panels)})
    run.save(str(path),base_path=str(output),policy='now')
    plot_report(report,output/'convergence.png')
    run.log({'convergence/figure':wandb.Image(str(output/'convergence.png'))})
    run.save(str(output/'convergence.png'),base_path=str(output),policy='now')
    run.summary.update(dict(phase='complete',samples=cfg['conditions']*max(cfg['prefixes']),
                       reused_samples=cfg['conditions']*cfg.get('inherited_candidates',0),
                       new_samples=cfg['conditions']*len(new_candidate_indices(cfg)),
                       inherited_prefixes_verified=bool(cfg.get('inherited_candidates')),policy_updates=0,classifier_fits=0,
                       report_path=str(path),samples_path=str(output/'samples_and_scores.npz')))
    (output/'COMPLETE').write_text('Inference and nested paired analysis complete\n')
    print('REPORT:',path,'WANDB:',run.url,flush=True)


def plot_report(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks=[int(k) for k in report['prefixes']]
    fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    for j,ax in enumerate(axs.flat):
        for arm,index in [('unweighted',0),('reweighted',1)]:
            y=np.array([report['prefixes'][str(k)][arm][j] for k in ks])
            ci=np.array([report['prefixes'][str(k)]['matrix_ci95'] for k in ks])[:,:,index,j]
            ax.plot(ks,y,'o-',label=arm)
            ax.fill_between(ks,ci[:,0],ci[:,1],alpha=.15)
        ax.axhline(report['truth'][j],color='black',linestyle='--',label='matched truth')
        ax.set_xscale('log',base=2)
        ax.set(title='C'+('k','r','n')[j//3]+('k','r','n')[j%3],xticks=ks,
               xticklabels=[str(k) for k in ks],xlabel='K (nested draws)')
    axs[0,0].legend()
    fig.suptitle('Frozen FiLM / raw step1110 — event-bootstrap pointwise 95% intervals')
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--analyze-only',type=Path,help='Completed sampling directory; no GPU or sampling rerun')
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args=p.parse_args();cfg=read_settings(args.config)
    if args.analyze_only:
        cfg=json.loads((args.analyze_only/'manifest.json').read_text())
        cfg['output']=str(args.analyze_only.resolve())
    else:
        cfg,inputs=prepare(cfg)
        cfg=validate_extension(cfg,inputs)
        print('READY:',dict(events=cfg['conditions'],workers=cfg['workers'],prefixes=cfg['prefixes'],
            classifier=cfg['classifier_run'],generator=cfg['generator_checkpoint']),flush=True)
        if args.prepare_only: return
        output=Path(cfg['output_root'])/('sampling-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
        cfg['output']=str(output)
        np.savez(output/'inputs.npz',**inputs); del inputs
        copy_inherited_shards(cfg)
        (output/'manifest.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    output=Path(cfg['output'])
    import wandb
    with wandb.init(entity=cfg['logger']['entity'],project=cfg['logger']['project'],name=cfg['logger']['name'],
        group=cfg['logger']['group'],tags=['inference-only','raw-1110','frozen-FiLM','16-gpu','nested-K'],
        config=cfg,dir=str(output),mode='online') as run:
        (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
        run.define_metric('candidates_per_condition')
        run.define_metric('convergence/*',step_metric='candidates_per_condition')
        run.define_metric('Cij/*',step_metric='candidates_per_condition')
        run.save(str(output/'manifest.json'),base_path=str(output),policy='now')
        try:
            if not args.analyze_only:
                import ray
                ray.init(address=args.ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
                    (str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),os.environ.get('PYTHONPATH','')))}})
                if ray.cluster_resources().get('GPU',0)<cfg['workers']: raise ValueError('Requires 16 Ray GPUs')
                reused=cfg['conditions']*cfg.get('inherited_candidates',0)
                run.summary.update(dict(phase='sampling',reused_samples=reused,
                    new_samples_requested=cfg['conditions']*len(new_candidate_indices(cfg))))
                remote=ray.remote(num_gpus=1,num_cpus=1,max_calls=1,max_retries=0)(worker)
                pending=[remote.remote(cfg,i) for i in range(cfg['workers'])]
                try:
                    while pending:
                        done,pending=ray.wait(pending,num_returns=1,timeout=10)
                        ray.get(done)
                        progress=[json.loads(f.read_text()) for f in output.glob('progress-*.json')]
                        covered={v['rank'] for v in progress}
                        # Ranks without a new completion still own their copied old draws.
                        complete=sum(v['samples'] for v in progress)+sum(
                            len(range(i,cfg['conditions'],cfg['workers']))*cfg.get('inherited_candidates',0)
                            for i in range(cfg['workers']) if i not in covered)
                        run.log({'sampling/samples_complete':complete,
                                 'sampling/new_samples_complete':complete-reused,
                                 'sampling/finished_workers':cfg['workers']-len(pending)})
                finally:
                    for ref in pending: ray.cancel(ref,force=True)
                    ray.shutdown()
            finish(cfg,run)
        except BaseException:
            run.summary['phase']='failed'
            raise


if __name__=='__main__': main()
