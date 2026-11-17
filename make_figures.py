"""Figures for the README: ECE by method, and a reliability diagram on unfamiliar tasks.

Run: python make_figures.py   -> figures/ece.png, figures/reliability.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from calibrate import conf_correct, fit_temperature, load
from data import FAMILIAR, UNFAMILIAR
from metrics import ece, reliability

SURFACE, INK, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#6b6a64', '#e6e5e0'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # fixed categorical order
MARKERS = ['o', 's', '^', 'D']
SFT_SEEDS = ['sft', 'sft_s1', 'sft_s2']

plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})


def spread_cc(tag, tasks):
    T = fit_temperature(load(tag, 'cal'))
    return conf_correct([r for r in load(tag, 'test') if r['task'] in tasks], T)


def written_cc(tag, tasks):
    rows = [json.loads(l) for l in open(f'results/written_{tag}_test.jsonl')]
    rows = [r for r in rows if r['task'] in tasks]
    return [r['conf'] for r in rows], [r['correct'] for r in rows]


METHODS = [('Base + temperature', lambda t: spread_cc('Qwen3-1.7B', t)),
           ('Fine-tuned + temperature', None),
           ('Base, written confidence', lambda t: written_cc('Qwen3-1.7B', t)),
           ('RL (Brier), written', lambda t: written_cc('rlw_brier', t))]


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ece_chart():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(METHODS))
    for j, (split, tasks) in enumerate([('Familiar tasks', FAMILIAR), ('Unfamiliar tasks', UNFAMILIAR)]):
        vals, errs = [], []
        for name, fn in METHODS:
            if fn is None:
                e = [ece(*spread_cc(s, tasks)) for s in SFT_SEEDS]
                vals.append(np.mean(e)); errs.append(np.std(e))
            else:
                vals.append(ece(*fn(tasks))); errs.append(0)
        bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.34, color=SERIES[j], label=split, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x + (j - 0.5) * 0.36, vals, yerr=[e if e else np.nan for e in errs], fmt='none', ecolor=INK, capsize=3, lw=1)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.006, f'{v:.3f}', ha='center', va='bottom', fontsize=9, color=MUTED)
    ax.set_xticks(x, [m[0].replace(', ', ',\n').replace(' + ', ' +\n') for m in METHODS])
    ax.set_ylabel('Calibration error (ECE, lower is better)')
    ax.set_title('Fine-tuning + temperature beats RL-trained confidence', loc='left', fontsize=12)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig('figures/ece.png', dpi=160)


def reliability_chart():
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--', label='Perfect calibration')
    for i, (name, fn) in enumerate(METHODS):
        conf, correct = spread_cc('sft', UNFAMILIAR) if fn is None else fn(UNFAMILIAR)
        pts = [(c, a) for _, n, c, a in reliability(conf, correct) if n >= 20]   # skip near-empty buckets
        ax.plot(*zip(*pts), color=SERIES[i], marker=MARKERS[i], markersize=7, lw=2, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel='Stated confidence', ylabel='Actual accuracy')
    ax.set_title('Unfamiliar tasks: points below the line are overconfident', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_aspect('equal')
    style(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig('figures/reliability.png', dpi=160)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    ece_chart()
    reliability_chart()
    print('wrote figures/ece.png, figures/reliability.png')
