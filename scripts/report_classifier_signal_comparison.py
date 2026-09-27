#!/usr/bin/env python3
"""Build presentation figures from saved classifier diagnostics only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/'artifacts/classifier_signal_comparison'


def read(path):
    return json.loads(path.read_text())


def main():
    arms = {}
    for name, run in [('Old / pretrain', 'oldpre01'), ('Pair token / step1110', 'h4pair01')]:
        p = ROOT/run
        arms[name] = dict(health=read(p/'ratio_health.json'),
                         topology=read(p/'pair_topology_analysis.json'),
                         features=read(p/'tail_attribution.json'))
    colors = ['#5576A6', '#CF7047']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    x = np.arange(3)
    for idx, (name, arm) in enumerate(arms.items()):
        h = arm['health']
        vals = [100*h[key] for key in ['raw/ess_fraction', 'raw/top1pct_mass', 'raw/max_weight_mass']]
        bars = axes[0, 0].bar(x+(idx-.5)*.36, vals, width=.36, color=colors[idx], label=name)
        axes[0, 0].bar_label(bars, fmt='%.2f', fontsize=9)
        w = arm['topology']['wasserstein1_radians']
        vals = [w[k]['raw']/w[k]['unweighted'] for k in ['acoplanarity', 'acollinearity']]
        bars = axes[0, 1].bar(np.arange(2)+(idx-.5)*.36, vals, width=.36, color=colors[idx])
        axes[0, 1].bar_label(bars, fmt='%.2f', fontsize=9)
        f = arm['features']
        keys = ['acoplanarity', 'acollinearity', 'visible_a_pt', 'visible_b_pt', 'visible_a_cos_theta', 'visible_b_cos_theta']
        axes[1, 0].barh(np.arange(len(keys))+(idx-.5)*.36, [f['spearman_logit'][k] for k in keys], height=.36, color=colors[idx])
        regions = [r for r in f['regions'] if r['region'] == 'acoplanarity' and r['threshold_radians'] >= 1e-4]
        axes[1, 1].plot([r['event_fraction']*100 for r in regions], [r['raw_weight_mass']*100 for r in regions], 'o-', color=colors[idx], label=name)
        for r in regions:
            axes[1, 1].annotate(f"<{r['threshold_radians']:g} rad", (r['event_fraction']*100, r['raw_weight_mass']*100), xytext=(4, 3), textcoords='offset points', fontsize=8, color=colors[idx])
    axes[0, 0].set(xticks=x, xticklabels=['ESS / N', 'Top 1% mass', 'Largest event mass'], ylabel='Percent', title='Raw ratio concentration')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].margins(y=.18)
    axes[0, 1].axhline(1, color='gray', ls='--', lw=1)
    axes[0, 1].set(xticks=[0, 1], xticklabels=['Acoplanarity', 'Acollinearity'], ylabel='Weighted W1 / original W1', title='Angular reweighting (lower is better)')
    axes[0, 1].margins(y=.18)
    axes[1, 0].set(yticks=np.arange(len(keys)), yticklabels=['Acoplanarity', 'Acollinearity', 'Visible A pT', 'Visible B pT', 'Visible A cos(theta)', 'Visible B cos(theta)'], xlabel='Spearman correlation with logit', title='What is associated with high scores?')
    axes[1, 0].axvline(0, color='gray', lw=1)
    axes[1, 1].plot([0, 100], [0, 100], '--', color='gray', lw=1)
    axes[1, 1].set(xlim=(0, 100), ylim=(0, 105), xlabel='Events below acoplanarity threshold (%)', ylabel='Raw weight mass in region (%)', title='Weight concentration near the angular endpoint')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Classifier signals: historical weak recipe vs pair token\nDifferent generator sources; descriptive comparison, not a causal architecture test', fontsize=13)
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT/f'classifier_signals.{ext}', dpi=180)
    plt.close(fig)
    snap = read(ROOT/'learning_snapshot.json')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    prefix = 'omnifold_live/raw_staleness_audit/'
    for idx, rid in enumerate(['oldpre01', 'h4pair01']):
        rows = next(r for r in snap['runs'] if r['id'] == rid)['history']
        step = [r['omnifold_live/meta/fit_step'] for r in rows]
        for ax, key in zip(axes[:2], ['classifier_fit/raw_staleness_audit/fit00001_g000000_i01_f00_r01/validation_loss', prefix+'validation_auc']):
            ax.plot(step, [r[key] for r in rows], color=colors[idx], label=list(arms)[idx])
        summary=next(r for r in snap['runs'] if r['id']==rid)['summary']
        bars=axes[2].bar(idx,100*summary[prefix+'gradient_clip_fraction'],color=colors[idx])
        axes[2].bar_label(bars,fmt='%.1f%%')
    for ax, title in zip(axes[:2], ['Validation BCE', 'Validation AUC']):
        ax.set(title=title, xlabel='Classifier optimizer updates')
        ax.spines[['top', 'right']].set_visible(False)
    axes[2].set(title='Updates triggering gradient clipping',xticks=[0,1],xticklabels=['Old','Pair token'],ylabel='Percent',ylim=(0,100))
    axes[2].spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Learning trajectories (last iterate curves; final evaluation restores best validation BCE)', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(ROOT/f'learning_comparison.{ext}', dpi=180)


if __name__ == '__main__':
    main()
