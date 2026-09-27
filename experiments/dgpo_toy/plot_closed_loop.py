"""Plot saved closed-loop evidence; never train or select a checkpoint."""
import argparse
import json
import os
from pathlib import Path

from .closed_loop_lab import toy_path


def plot(output):
    output = toy_path(output)
    report = json.loads((output/'report.json').read_text())
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fits = {'classifier': report['fit']} if 'fit' in report else report['audits']
    for step, fit in fits.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name in fit['test']:
            rows = [row for row in fit['history'] if row['arm'] == name]
            for ax, key in zip(axes, ('validation_bce', 'validation_auc')):
                ax.plot([r['step'] for r in rows], [r[key] for r in rows], label=name, lw=1)
            selected = next(r for r in rows if r['step'] == fit['test'][name]['selected_step'])
            axes[0].scatter([selected['step']], [selected['validation_bce']], s=18)
        for ax, label in zip(axes, ('Validation BCE', 'Validation AUC')):
            ax.set(xlabel='Classifier optimizer updates', ylabel=label)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Round {report['plan']['round']}: cold classifiers | {step}")
        fig.tight_layout()
        fig.savefig(output/f'fit_curves_{step}.png', dpi=150)
        plt.close(fig)
    if report['plan']['kind'] == 'classifier':
        datasets = list(report['plan']['datasets'])
        if len(datasets) > 1 and all(name.startswith('step') for name in datasets):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for feature in report['plan']['features']:
                points = [(int(d[4:]), report['fit']['test'][f'{d}__{feature}']) for d in datasets]
                points.sort(key=lambda p: p[0])
                for ax, metric in zip(axes, ('auc', 'bce')):
                    ax.plot([p[0] for p in points], [p[1][metric] if p[1]['valid'] else float('nan') for p in points],
                            'o-', label=feature)
                    ax.set(xlabel='DGPO updates', ylabel=f'Fresh test {metric.upper()}')
                    ax.grid(alpha=.2)
            axes[0].legend()
            fig.suptitle('Cold independent fits at each checkpoint; lines do not imply monotonicity between points')
            fig.tight_layout()
            fig.savefig(output/'fresh_audit_trajectory.png', dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    plot(parser.parse_args().output)
