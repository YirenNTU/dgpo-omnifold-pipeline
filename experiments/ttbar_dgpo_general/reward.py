"""General probability mixture, in native (K,B) order; no additive rewards."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from experiments.ttbar_general_classifier.core import GeneralClassifier


def geometry_tensor(x):
    # Match the saved-cache float64 geometry calculation, then cast to float32.
    x=x.double();eta,phi=x[...,1],x[...,2]
    direct=torch.stack((x[...,0],eta,phi.sin(),phi.cos()),-1).flatten(-2)
    u=torch.stack((phi.cos()/eta.cosh(),phi.sin()/eta.cosh(),eta.tanh()),-1)
    joint=(u[...,0,:,None]*u[...,1,None,:]).flatten(-2)
    return torch.cat((direct,u.flatten(-2),joint),-1).float()


def nu_p4_tensor(x):
    x=x.double();pt=x[...,0].expm1();eta=x[...,1];phi=x[...,2]
    return torch.stack((pt*eta.cosh(),pt*phi.cos(),pt*phi.sin(),pt*eta.sinh()),-1)


def log_mixture(logits, candidates, truth, scale, alpha=.3, temperature=1.):
    if logits.ndim!=2 or candidates.shape[:2]!=logits.shape or candidates.shape[2:]!=(2,3):
        raise ValueError('Require (K,B) logits and (K,B,2,3) native coordinates')
    if logits.shape[0]<2 or not 0<=alpha<1 or temperature<=0:
        raise ValueError('Require K>=2, 0<=alpha<1, T>0')
    scale=torch.as_tensor(scale,device=logits.device,dtype=torch.float64)
    if scale.shape!=(4,) or not torch.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid fixed training-only p4 scales')
    # SAME reconstructed visible on both sides cancels exactly in p4 distance.
    # Scales still come from reconstructed-visible + MC-nu TOP p4, not neutrino RMS.
    delta=nu_p4_tensor(candidates)-nu_p4_tensor(truth)[None]
    distance=(delta/scale).square().mean((-1,-2))
    if not torch.isfinite(logits).all() or not torch.isfinite(distance).all():
        raise ValueError('Nonfinite classifier/oracle input; no clipping or replacement')
    logq=torch.log_softmax(logits/temperature,dim=0)
    winner=distance.argmin(0,keepdim=True)
    oracle=torch.full_like(logq,-torch.inf)
    oracle.scatter_(0,winner,0.)
    logw=logq if alpha==0 else torch.logaddexp(logq+math.log1p(-alpha),oracle+math.log(alpha))
    return logw,logq,distance


@torch.no_grad()
def condition_tensor(model,batch):
    xm=batch['x_mask'][...,None];gm=batch['conditions_mask'][...,None]
    x=model.sequential_normalizer(batch['x'],mask=xm)
    g=model.global_normalizer(batch['conditions'][:,None],mask=gm)
    return torch.cat((x.flatten(1),xm.flatten(1),g.flatten(1),gm.flatten(1)),1)


@torch.no_grad()
def features_tensor(model,batch,candidate):
    mask=torch.ones((*candidate.shape[:2],1),device=candidate.device)
    normalized=model.invisible_normalizer(candidate,mask=mask)
    captured=[]
    hook=model.TruthGeneration.generator.register_forward_pre_hook(
        lambda module,args:captured.append(args[0][:,-2:].detach()))
    observed={k:v for k,v in batch.items() if k in ('x','x_mask','conditions','conditions_mask','classification','num_sequential_vectors')}
    try:
        model.predict_diffusion_vector(normalized,observed,torch.zeros(len(candidate),device=candidate.device),'neutrino',noise_mask=mask)
    finally:hook.remove()
    if len(captured)!=1:raise RuntimeError('Expected one clean pre-velocity feature capture')
    return torch.cat((captured[0].flatten(1),geometry_tensor(candidate)),1)


def load_feature_model(directory,device):
    # Fresh Config instance: never mutate the DGPO global_config during refits.
    from RL.DGPO_neutrino.model_utils import load_training_config,build_evenet_on_device,load_normalization_dict
    from evenet.network.loss.famo import FAMO
    directory=Path(directory);cfg=load_training_config(directory/'policy.yaml')
    model=build_evenet_on_device(cfg,load_normalization_dict(cfg),device)
    f=cfg.options.Training.FAMO
    tasks=f.get('detailed_loss_list') if f.get('turn_on',False) and f.get('detailed_loss',False) else ['classification','regression','assignment','generation','segmentation']
    model.register_module('famo',FAMO(task_list=tasks,device=device,lr=f.get('lr',.025),turn_on=f.get('turn_on',False),logits_bound=f.get('logits_bound',1.)))
    state=torch.load(directory/'policy.ckpt',map_location='cpu',weights_only=False)['state_dict']
    clean={k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')}
    model.load_state_dict(clean or state,strict=True)
    return model.eval().requires_grad_(False)


class GeneralOracleReward:
    name='general_top_oracle_log_mixture'
    def __init__(self,feature_model,payload,*,alpha=.3,temperature=1.,feature_batch_size=1024):
        self.feature_model=feature_model.eval().requires_grad_(False)
        self.device=next(feature_model.parameters()).device
        self.alpha,self.temperature=alpha,temperature
        self.feature_batch_size=int(feature_batch_size)
        self.stats=payload['stats'];self.head_config=payload['config']['head']
        if self.stats['oracle_reference']!='reco_top_truthnu_p4':raise ValueError('Wrong oracle target')
        self.head=GeneralClassifier(self.stats['condition_dim'],self.stats['candidate_dim'],**self.head_config).to(self.device)
        self.install(payload['state_dict'])
        self.mean=self.stats['mean'].to(self.device);self.scale=self.stats['scale'].to(self.device)
        self.last={};self.validation_sink=None

    def install(self,state):
        self.head.load_state_dict(state,strict=True);self.head.eval().requires_grad_(False)

    @torch.no_grad()
    def compute(self,candidates,batch,mask=None):
        k,b=candidates.shape[:2]
        if not torch.isfinite(candidates).all():raise ValueError('Nonfinite rollout')
        if not bool(batch['x_invisible_mask'].bool().all()):raise ValueError('Require both valid neutrino targets')
        if not bool((batch['classification']==0).all()):raise ValueError('Expected MG5 TTbar class0')
        cond=condition_tensor(self.feature_model,batch)
        # Candidate chunks cap the feature-only forward at 1024; rollout/event batch
        # and DGPO gradients are unchanged. No repeated CPU normalization per K.
        logits=[];kc=max(1,self.feature_batch_size//b)
        for start in range(0,k,kc):
            x=candidates[start:start+kc];n=len(x)
            repeated={key:v.unsqueeze(0).expand(n,*v.shape).reshape(n*b,*v.shape[1:])
                for key,v in batch.items() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==b}
            f=features_tensor(self.feature_model,repeated,x.reshape(n*b,2,3))
            labels=batch['classification'].reshape(-1).repeat(n).long()
            logits.append(self.head(cond.repeat(n,1),(f-self.mean)/self.scale,labels).reshape(n,b))
        logits=torch.cat(logits)
        logw,logq,d=log_mixture(logits,candidates,batch['x_invisible'],self.stats['oracle_scale'],self.alpha,self.temperature)
        self.last=dict(logits=logits,logw=logw,logq=logq,distance=d)
        if self.validation_sink is not None:self.validation_sink(candidates,batch,self.last)
        return logw

    def metrics(self):
        w=self.last['logw'].exp();q=self.last['logq'].exp()
        return {'reward/mixture_ess':float((1/w.square().sum(0)).mean()),
                'reward/general_ess':float((1/q.square().sum(0)).mean()),
                'reward/mixture_max_weight':float(w.max(0).values.mean()),
                'reward/oracle_fraction':self.alpha}
