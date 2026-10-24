"""Energy-free tau directions and reconstructed visible inputs for pair interactions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_DIM=24


def feature_spec():
    return dict(version=1,dimension=FEATURE_DIM,leg_dimension=12,leg_width=64,bilinear_outputs=16,
                shared_leg_encoder=True,shared_leg_film=True,context_width=64,
                features=['delta_theta/0.05','sin(delta_phi)/0.05','(cos(delta_phi)-1)/0.0025',
                          'visible_unit_xyz','tau_unit_xyz','log1p_visible_E','log1p_visible_p','log1p_visible_mass'],
                zero_output=True,general_head_trainable=False,cross_attention=False,tau_energy_input=False)


def pair_features(visible_a,visible_b,deltas):
    d=np.asarray(deltas,float)
    if d.ndim!=3 or d.shape[1:]!=(2,2) or not np.isfinite(d).all():raise ValueError('Invalid candidate offsets')
    blocks=[]
    for i,visible in enumerate((visible_a,visible_b)):
        v=np.asarray(visible,float)
        if v.shape!=(len(d),4) or not np.isfinite(v).all() or (v[:,0]<0).any():raise ValueError('Invalid reconstructed visible p4')
        p=np.linalg.norm(v[:,1:],axis=1)
        if (p<=0).any():raise ValueError('Undefined visible direction')
        unit=v[:,1:]/p[:,None]
        theta=np.arctan2(np.linalg.norm(v[:,1:3],axis=1),v[:,3])+d[:,i,0]
        phi=np.arctan2(v[:,2],v[:,1])+d[:,i,1]
        tau=np.stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)),axis=1)
        # Reconstructed visible mass only; no tau mass-shell/energy assumption.
        mass=np.sqrt(np.maximum(v[:,0]**2-p*p,0))
        local=np.stack((d[:,i,0]/.05,np.sin(d[:,i,1])/.05,(np.cos(d[:,i,1])-1)/.0025),axis=1)
        blocks.append(np.c_[local,unit,tau,np.log1p(v[:,0]),np.log1p(p),np.log1p(mass)])
    result=np.concatenate(blocks,axis=1).astype('float32')
    if result.shape!=(len(d),FEATURE_DIM) or not np.isfinite(result).all():raise ValueError('Invalid pair features')
    return result


class FrozenGeneralPairInteraction(nn.Module):
    def __init__(self,base,dimension):
        super().__init__()
        from scripts.tau_conditioning_heads import ConditioningBlock
        self.base=base.eval().requires_grad_(False);self.base_candidate_dim=dimension
        self.condition=nn.Sequential(nn.Linear(base.condition_encoder[0].in_features,64),nn.SiLU())
        self.label=nn.Linear(base.num_process_classes,64,bias=False)
        self.leg_encoder=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.leg_film=ConditioningBlock('film',64,0.,64)
        self.leg_norm=nn.LayerNorm(64)
        self.bilinear=nn.Bilinear(64,64,16,bias=False)
        self.pair_encoder=nn.Sequential(nn.Linear(208,64),nn.SiLU())
        self.readout=nn.Linear(64,1)
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,condition,candidate):
        from scripts.tau_bounded_ratio import bounded_log_ratio
        if candidate.shape[1]!=self.base_candidate_dim+FEATURE_DIM:raise ValueError('Wrong pair feature layout')
        general=self.base(condition,candidate[:,:self.base_candidate_dim])
        labels=condition[:,self.base.process_label_index].long()
        context=self.condition(condition[:,self.base.continuous_condition_columns])
        context=context+self.label(F.one_hot(labels,self.base.num_process_classes).to(context.dtype))
        legs=candidate[:,self.base_candidate_dim:].reshape(-1,2,12)
        h=self.leg_encoder(legs)
        h=self.leg_norm(h+self.leg_film(h,context[:,None,:]))
        a,b=h[:,0],h[:,1]
        joint=torch.cat((a,b,a*b,self.bilinear(a,b)/8),dim=1)
        correction=self.readout(self.pair_encoder(joint)).flatten()
        return bounded_log_ratio(general+correction,30)
