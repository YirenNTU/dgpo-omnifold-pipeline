"""Plot completed local retention rounds and optionally archive offline W&B logs."""
import argparse
import json
from pathlib import Path
import numpy as np
from .closed_loop_lab import toy_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--offline-wandb',action='store_true')
    args=parser.parse_args();root=toy_path(args.directory)
    old=json.loads((root/'round01/report.json').read_text())
    late=json.loads((root/'round03/measurements/report.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for arm,color in (('raw','#52657d'),('fourier','#137b74')):
        old_values=old['arms'][f'round04/{arm}']['fixed_horizon_means']
        steps=np.array([0,25,100,300,1000])
        vals=np.array([old_values[str(s)] for s in steps])-old_values['0']
        axes[0].plot(steps,vals,'o-',color=color,label=arm+' FiLM')
        metrics=late['anchors'][arm]['endpoint_gains']
        steps=np.array([0,1,5,20]); vals=np.array([metrics[str(s)]['mean'] for s in steps])
        error=np.array([[metrics[str(s)][key] for s in steps] for key in ('lo95','hi95')])
        axes[1].errorbar(steps,vals,yerr=np.array([vals-error[0],error[1]-vals]),
                         fmt='o-',capsize=3,color=color,label=arm+' FiLM')
    for ax in axes:
        ax.axhline(0,color='black',linewidth=.7);ax.grid(alpha=.15);ax.legend(frameon=False)
        ax.set_ylabel('Held-out fixed reward gain (all candidates)')
    axes[0].set_title('Saved trajectory: broad improvement')
    axes[0].set_xlabel('DGPO updates from original toy source')
    axes[1].set_title('Native continuation from step 1000')
    axes[1].set_xlabel('Additional DGPO updates; pointwise 95% intervals')
    fig.suptitle('No large aggregate reward-loss failure reproduced in this toy window',fontsize=12)
    fig.savefig(root/'reward_retention.png',dpi=180)
    plt.close(fig)
    if args.offline_wandb:
        import wandb
        with wandb.init(project='dgpo-toy-local',mode='offline',dir=str(root),
                name='Do late updates retain reward? | conditional toy | completed-result replay',
                group='Conditional reward retention',tags=['toy-only','posthoc-import','no-production-change'],
                config={'logging':'POST-HOC IMPORT of completed calculations, not live training',
                        'source':str(root),'trained_updates_per_arm':20,'velocity_coefficient':1.},
                settings=wandb.Settings(disable_git=True)) as run:
            for step in (0,1,5,20):
                row={'relative_step':step}
                for arm in ('raw','fourier'):
                    row.update({f'{arm}/reward_gain/{k}':v for k,v in late['anchors'][arm]['endpoint_gains'][str(step)].items()})
                run.log(row)
            run.summary['failure_reproduced']=False
            run.summary['fresh_classifier_fitted']=False
            metadata={'id':run.id,'name':run.name,'directory':run.dir,'mode':'offline','posthoc_import':True}
            from .truth_pretrain import atomic_json
            atomic_json(root/'offline_wandb.json',metadata)
    print(root/'reward_retention.png')


if __name__=='__main__':main()
