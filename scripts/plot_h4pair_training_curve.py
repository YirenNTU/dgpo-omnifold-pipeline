"""Plot h4pair01 recorded fit losses, with the restored checkpoint marked."""
from pathlib import Path
import json,csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('artifacts/classifier_signal_comparison/h4pair01/training_curve')
h=json.loads((out/'wandb_history.json').read_text());p=h['prefix']
b=torch.load('/tmp/h4pair01_best_classifier_and_test.pt',map_location='cpu',weights_only=False)
d=b['fit_diagnostics'];val={int(r['step']):float(r['loss']) for r in d['validation_history']}
train={int(r[p+'step']):float(r[p+'training_loss']) for r in h['rows'] if r.get(p+'training_loss') is not None}
for r in h['rows']:
 if r.get(p+'validation_loss') is not None:
  assert np.isclose(val[int(r[p+'step'])],r[p+'validation_loss'],atol=1e-7)
assert max(train)==d['steps_completed']==1048
best=d['best_step'];assert best==min(val,key=val.get)
with (out/'loss_curves.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['classifier_update','training_bce','validation_bce'])
 for step in sorted(set(train)|set(val)):w.writerow([step,train.get(step,''),val.get(step,'')])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.labelsize':18})
fig,ax=plt.subplots(figsize=(16,6.7))
ax.plot(list(train),list(train.values()),color='#397FC0',lw=2.2,label='Training BCE (logged batches)')
ax.plot(list(val),list(val.values()),color='#C66420',lw=2.2,label='Validation BCE')
ax.axhline(np.log(2),color='#89939D',lw=1,ls=':')
ax.text(30,.698,'Uninformative classifier: ln 2',color='#67727D',fontsize=12)
ax.axvline(best,color='#78838E',ls='--',lw=1)
ax.scatter([best],[val[best]],color='#C66420',s=55,zorder=4)
ax.annotate('Selected checkpoint\nUpdate 1,008 · validation BCE 0.3787',xy=(best,val[best]),xytext=(610,.475),fontsize=15,arrowprops={'arrowstyle':'->','color':'#56616C'},color='#202833')
ax.set(xlabel='Classifier optimizer update',ylabel='Binary cross-entropy',xlim=(0,1080),ylim=(.34,.725))
ax.set_xticks([0,200,400,600,800,1000]);ax.grid(alpha=.18)
ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper right',frameon=False,fontsize=15)
fig.subplots_adjust(left=.075,right=.98,bottom=.15,top=.97)
fig.savefig(out/'h4pair01_training_validation.png',dpi=200)
fig.savefig(out/'h4pair01_training_validation.pdf')
(out/'README.md').write_text('Run: https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/h4pair01\nState: finished. Train: 110 unsmoothed logged batch BCE values from exact W&B scan_history, using classifier_fit/.../step as x. Validation: all 263 evaluations from saved best-classifier bundle, including initialization; all overlapping W&B validation entries agree within 1e-7. No interpolation of missing measurements into CSV. Lines connect observed points. Selected update 1008, BCE 0.3786752820. Fit stopped at 1048, final validation BCE 0.3848031163. Source generator: c4a91e07 policy step1110; that is not the classifier update clock.\n')
print('Verified train/validation alignment; selected:',best,val[best])
