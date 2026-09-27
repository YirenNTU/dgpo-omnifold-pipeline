"""Ordered truth/generated/reweighted kinematics from the saved step1110 export."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from diagnose_h4_ratio_tail import unpack
from diagnose_h4_topology_resolution import reconstruct

def plot_label(key):
 """Short physical labels, independent of storage-field names."""
 pair = {
  'acoplanarity': ('Acoplanarity', r'$\pi-|\Delta\phi_{AB}|$ [rad]'),
  'acollinearity': ('Acollinearity', r'$\pi-\psi_{AB}$ [rad]'),
  'opening': ('Opening angle', r'$\psi_{AB}$ [rad]'),
  'pair_phi_residual': ('Azimuthal residual', r'$\mathrm{wrap}(\phi_A-\phi_B-\pi)$ [rad]'),
  'pair_theta_sum_residual': ('Polar-angle residual', r'$\theta_A+\theta_B-\pi$ [rad]'),
  'pair_cos_opening': ('Opening-angle cosine', r'$\cos\psi_{AB}$'),
 }
 if key in pair:return pair[key]
 visible=key.startswith('visible_')
 leg,name=(key[len('visible_'):] if visible else key).split('_',1)
 symbols={
  'delta_theta':r'\Delta\theta', 'delta_phi':r'\Delta\phi',
  'theta':r'\theta', 'phi':r'\phi', 'eta':r'\eta',
  'px':'p_x','py':'p_y','pz':'p_z','pt':'p_T','p':r'|\mathbf{p}|',
  'cos_theta':r'\cos\theta','sin_phi':r'\sin\phi','cos_phi':r'\cos\phi',
  'direction_x':r'\hat{n}_x','direction_y':r'\hat{n}_y',
  'visible_opening':r'\angle(\tau,\mathrm{vis})',
 }
 unit=' [rad]' if name in ['delta_theta','delta_phi','theta','phi','visible_opening'] else ''
 if name in ['px','py','pz','pt','p']:unit=' [saved units]'
 return (('Visible ' if visible else r'$\tau$ ')+leg, '$'+symbols[name]+'$'+unit)

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',default='/tmp/h4pair01_best_classifier_and_test.pt');p.add_argument('--output',default='artifacts/classifier_signal_comparison/step1110_kinematics');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 b=torch.load(a.bundle,map_location='cpu',weights_only=False);f=unpack(b);n=len(b['test_truth'])
 w=torch.softmax(b['gen_logits'].double().reshape(-1),0).numpy();flat=np.ones(n)/n
 wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
 datasets=[];labels={};groups=[[],[],[],[],[],[],[]]
 for key in ['test_truth','test_generated']:
  z=b[key].double().numpy().reshape(n,2,2);d={};angs=[]
  for j,leg in enumerate(['A','B']):
   v=np.stack([f[f'lead_{leg.lower()}_visible_{x}'].double().numpy() for x in ['px','py','pz']],1)
   mag=np.linalg.norm(v,axis=1);pt=np.hypot(v[:,0],v[:,1]);tv=np.arctan2(pt,v[:,2]);pv=np.arctan2(v[:,1],v[:,0])
   th=tv+z[:,j,0];ph=wrap(pv+z[:,j,1]);angs.append((th,ph))
   vals={'delta_theta':z[:,j,0],'delta_phi':wrap(z[:,j,1]),'theta':th,'phi':ph,
    'cos_theta':np.cos(th),'sin_phi':np.sin(ph),'cos_phi':np.cos(ph),
    'direction_x':np.sin(th)*np.cos(ph),'direction_y':np.sin(th)*np.sin(ph)}
   u=np.stack([vals['direction_x'],vals['direction_y'],vals['cos_theta']],1)
   vals['visible_opening']=np.arctan2(np.linalg.norm(np.cross(u,v/mag[:,None]),axis=1),(u*v/mag[:,None]).sum(1))
   for name,x in vals.items():d[f'{leg}_{name}']=x
   for name,x in dict(px=v[:,0],py=v[:,1],pz=v[:,2],pt=pt,p=mag,theta=tv,phi=pv,eta=np.arcsinh(v[:,2]/pt)).items():d[f'visible_{leg}_{name}']=x
  top=reconstruct(f,b[key]);d.update({k:top[k] for k in ['acoplanarity','acollinearity','opening']})
  d['pair_phi_residual']=wrap(angs[0][1]-angs[1][1]-np.pi);d['pair_theta_sum_residual']=angs[0][0]+angs[1][0]-np.pi;d['pair_cos_opening']=top['topology'][:,2]
  datasets.append(d)
 T,G=datasets
 pages=[
 ('01_targets','Direct predictions: angular offsets',[f'{l}_{k}' for l in ['A','B'] for k in ['delta_theta','delta_phi']]),
 ('05_pair_geometry','Tau-pair structure',['acoplanarity','acollinearity','pair_theta_sum_residual']),
 ('08_visible_summary','Reweighting check: shared visible inputs',[f'visible_{l}_{k}' for l in ['A','B'] for k in ['pt','eta','phi']]),
 ]
 metrics={};manifest=[]
 plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':10})
 with PdfPages(out/'step1110_all_available_kinematics.pdf') as pdf:
  for stem,title,keys in pages:
   for chunk in range(0,len(keys),6):
    ks=keys[chunk:chunk+6];cols=3 if len(ks)==3 or len(ks)>4 else 2;rows=(len(ks)+cols-1)//cols
    fig=plt.figure(figsize=(cols*4.8,rows*4.4+1.5));grid=fig.add_gridspec(rows,cols,hspace=.55,wspace=.32)
    for i,k in enumerate(ks):
     row,col=divmod(i,cols);inner=grid[row,col].subgridspec(2,1,height_ratios=[3,1],hspace=.08);ax=fig.add_subplot(inner[0]);rat=fig.add_subplot(inner[1],sharex=ax)
     logx=k in ['acoplanarity','acollinearity']
     lo,hi=np.quantile(np.r_[T[k],G[k]],[.001,.999])
     if logx:lo=1e-6;hi=max(hi,.003);bins=np.geomspace(lo,hi,43)
     else:bins=np.linspace(lo,hi,43)
     h=[];v=[];coverage=[]
     for x,ww,color,label,style in [(T[k],flat,'#202833','Truth','-'),(G[k],flat,'#397FC0','Generated (1110)','--'),(G[k],w,'#C66420','Reweighted (H4)','-')]:
      hh=np.histogram(x,bins,weights=ww)[0];vv=np.histogram(x,bins,weights=ww**2)[0];h.append(hh);v.append(vv);coverage.append(float(hh.sum()))
      ax.stairs(100*hh,bins,color=color,label=label,linestyle=style,lw=1.7)
     mid=np.sqrt(bins[:-1]*bins[1:]) if logx else (bins[:-1]+bins[1:])/2
     valid=h[0]>=20/n
     for j,color in [(1,'#397FC0'),(2,'#C66420')]:
      r=np.divide(h[j],h[0],out=np.full_like(h[0],np.nan),where=valid)
      # Ratio points only: shared event correlations prevent naive independent errors.
      rat.plot(mid,r,color=color,lw=1.1,linestyle='--' if j==1 else '-')
     rat.axhline(1,color='#6A7077',lw=.8);rat.set_ylim(0,2);rat.set_ylabel('/ Truth');rat.grid(alpha=.2)
     panel_title,xlabel=plot_label(k)
     ax.set_title(panel_title);ax.set_ylabel('Events [% / bin]');ax.grid(alpha=.18);ax.tick_params(labelbottom=False)
     if logx:ax.set_xscale('log');rat.set_xscale('log')
     rat.set_xlabel(xlabel)
     sw=np.std(T[k]);ra=wasserstein_distance(T[k],G[k]);rw=wasserstein_distance(T[k],G[k],v_weights=w)
     metrics[k]={'raw_w1':ra,'weighted_w1':rw,'truth_std':sw,'raw_w1_over_std':ra/sw if sw else None,'weighted_w1_over_std':rw/sw if sw else None,'displayed_mass':dict(zip(['truth','generated','reweighted'],coverage))}
    handles,leglabels=fig.axes[0].get_legend_handles_labels();fig.legend(handles,leglabels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle(title,fontsize=15,y=.995)
    notes={
     '01_targets':r'$\Delta\theta=\theta_\tau-\theta_\mathrm{vis}$;  $\Delta\phi=\mathrm{wrap}(\phi_\tau-\phi_\mathrm{vis})$  |  A, B: two tau decay sides',
     '02_tau_directions':r'$\theta_\tau=\theta_\mathrm{vis}+\Delta\theta$;  $\phi_\tau=\mathrm{wrap}(\phi_\mathrm{vis}+\Delta\phi)$',
     '03_direction_components':r'$\hat{n}$: tau unit direction  |  Derived from predicted angles and visible inputs',
     '04_local_geometry':'Angle between each tau direction and its visible decay system',
     '05_pair_geometry':r'$\psi_{AB}$: 3D opening angle  |  $\Delta\phi_{AB}=\mathrm{wrap}(\phi_A-\phi_B)$  |  Acoplanarity / acollinearity: log axis, start at $10^{-6}$ rad',
     '06_visible_momenta':'Shared inputs: Truth and Generated overlap exactly; reweighting changes event weights',
     '08_visible_summary':'Shared inputs: Truth and Generated overlap exactly; reweighting changes event weights',
     '07_visible_angles':'Shared inputs: Truth and Generated overlap exactly; reweighting changes event weights',
    }
    footer=notes[stem]+'\n23,927 events  |  Full-sample normalization  |  Display tails omitted; ratio range 0–2  |  Details: README'
    fig.text(.5,.016,footer,ha='center',va='bottom',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.21 if rows==1 else .14)
    filename=stem+(f'_{chunk//6+1}' if len(keys)>6 else '')+'.png';fig.savefig(out/filename,dpi=180);pdf.savefig(fig);plt.close(fig);manifest.append(filename)
 result={'source_bundle':str(Path(a.bundle).resolve()),'generator':'c4a91e07 step1110','classifier':'h4pair01 restored best','N':n,'ESS':float(1/(w*w).sum()),'metrics':metrics,'pages':manifest,'scope':'13 selected angular/visible kinematics on 3 pages; no inferred neutrino/tau energies, masses or transverse momenta.'}
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps({'n':n,'ess':result['ESS'],'variables':len(metrics),'pages':manifest},indent=2))
if __name__=='__main__':main()
