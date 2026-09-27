"""Plot saved relation-access outputs only; no training or checkpoint selection."""
import argparse
import json
import os
from pathlib import Path

from .closed_loop_lab import toy_path


def plot(output):
    output = toy_path(output)
    report = json.loads((output/'report.json').read_text())
    plan = json.loads((output/'plan.json').read_text())
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    fits = {'classifier':report['fit']} if 'fit' in report else report['audits']
    for step,fit in fits.items():
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for name,stats in fit['test'].items():
            rows = [r for r in fit['history'] if r['arm']==name]
            xx = [r['step'] for r in rows]
            line, = axes[0].plot(xx,[r['validation_bce'] for r in rows],label=name)
            axes[0].plot(xx,[r['train_bce'] for r in rows],alpha=.25,color=line.get_color(),lw=.7)
            axes[1].plot(xx,[r['validation_auc'] for r in rows],label=name)
            selected = next(r for r in rows if r['step']==stats['selected_step'])
            axes[0].scatter([selected['step']],[selected['validation_bce']],color=line.get_color(),s=20)
        for ax,label in zip(axes,('BCE: validation / faint training minibatch','Validation AUC')):
            ax.set(xlabel='Classifier optimizer updates',ylabel=label); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Fresh classifier fits | '+str(step))
        fig.tight_layout()
        path = output/f'fit_curves_{step}.png'; fig.savefig(path,dpi=140); plt.close(fig)
        paths.append(str(path))
    if plan['kind']=='rl':
        fig,axes = plt.subplots(2,2,figsize=(11,8))
        for name in sorted({k.rsplit('_',1)[0] for k in report['points'] if k!='baseline_0'}):
            steps = [0,*plan['milestones']]
            points = [report['points']['baseline_0']]+[report['points'][f'{name}_{s}'] for s in steps[1:]]
            axes[0,0].plot(steps,[v.get('reward_gain',0) for v in points],'o-',label=name)
            axes[0,1].plot(steps,[v.get('conditioning',{}).get('fixed_reference_velocity_mse',0) for v in points],'o-',label=name)
            audits = [report['audits'][str(s)]['test'][f"{'baseline' if s==0 else name}__relative"] for s in steps]
            axes[1,0].plot(steps,[v['auc'] if v['valid'] else float('nan') for v in audits],'o-',label=name)
            axes[1,1].plot(steps,[v['structure']['order3_l2_debiased'] for v in points],'o-',label=name)
        for ax,label in zip(axes.flatten(),('Held-out mean fixed reward gain','Reference velocity MSE',
            'Fresh test AUC (invalid fits omitted)','Debiased order3 probability error')):
            ax.set(xlabel='Policy optimizer updates',ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('Sparse evaluation points; lines do not imply monotonic changes between audits')
        fig.tight_layout()
        path = output/'policy_curves.png'; fig.savefig(path,dpi=140); plt.close(fig); paths.append(str(path))
    return paths


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print('\n'.join(plot(parser.parse_args().output)))
