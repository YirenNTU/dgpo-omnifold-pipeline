"""Render recorded validation histories; no interpolation or training runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1')
report=json.loads((root/'report.json').read_text())
rows=report['actual_classifier']['history']
x=np.array([r['step'] for r in rows])
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
colors={'plain':'#2473b7','fourier':'#df7831'}
for name,color in colors.items():
    bce=np.array([r['arms'][name]['bce'] for r in rows])
    auc=np.array([r['arms'][name]['auc'] for r in rows])
    axs[0,0].plot(x,bce,color=color,label=name.title(),lw=1.5)
    axs[0,1].plot(x,auc,color=color,label=name.title(),lw=1.5)
    if name=='plain':
        mask=x>=8000
        axs[1,0].plot(x[mask],bce[mask],color=color,label='Validation BCE',alpha=.65)
        axs[1,0].plot(x[mask],np.minimum.accumulate(bce)[mask],color='#122a42',label='Best so far',lw=2)
        best=int(x[bce.argmin()])
        axs[1,0].scatter([best],[bce.min()],color='black',s=35,zorder=5)
        print('Plain best validation BCE by step:',json.dumps({str(s):float(bce[x<=s].min()) for s in [6000,8000,10000,12000,14000,16000]}))
for ax,title,ylabel in [(axs[0,0],'Actual diffusion vs truth: validation BCE','BCE (lower is better)'),
                         (axs[0,1],'Same held-out validation panel','AUC'),
                         (axs[1,0],'Plain: late-window zoom (note the small y range)','Validation BCE')]:
    ax.set(title=title,xlabel='Classifier optimizer steps',ylabel=ylabel)
    ax.grid(alpha=.2);ax.legend()
axs[0,0].axhline(np.log(2),color='gray',ls=':',label='Chance')
axs[0,1].axhline(.5,color='gray',ls=':')
c=np.linspace(-1,1,2001)
mu=np.array([-.77,-.31,.12,.63]);a=np.array([1.,-.85,.95,-1.])
g=np.tanh(2*(np.exp(-.5*((c[:,None]-mu)/.06)**2)*a).sum(1))
axs[1,1].plot(c,.5+.4*g,color='#71439b',lw=2,label='Truth: 0.5 + 0.4 g(c)')
axs[1,1].axhline(.5,color='gray',ls='--',label='Ideal reference: 0.5')
axs[1,1].set(title='Nonperiodic signed Gaussian bumps (NOT sine)',xlabel='Observed condition c',
             ylabel='P(positive triple parity | c)',ylim=(0,1))
axs[1,1].grid(alpha=.2);axs[1,1].legend()
fig.suptitle('Nonperiodic cube classifier — recorded validation, every 100 updates\nNo training-loss series was recorded; test data are not used in these curves',fontsize=13)
fig.savefig(root/'classifier_curves.png',dpi=170)
plt.close(fig)
