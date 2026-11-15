"""Temperature scaling on saved read-the-spread results.

Fits ONE global T on the familiar calibration split (min cross-entropy), then reports
test metrics before/after for familiar and unfamiliar tasks. No model re-run needed:
dividing saved log-probs by T and re-normalizing == dividing the raw logits by T
(softmax ignores a constant added to every option).

Run: python calibrate.py [model]       e.g. python calibrate.py Qwen3-1.7B
"""
import json
import sys

import numpy as np

from data import FAMILIAR, UNFAMILIAR
from metrics import reliability, summary


def load(model, split):
    return [json.loads(l) for l in open(f'results/spread_{model}_{split}.jsonl')]


def scaled_probs(logp, T):
    z = np.asarray(logp) / T
    z = z - z.max()                       # numerical safety; doesn't change softmax
    p = np.exp(z)
    return p / p.sum()


def conf_correct(rows, T=1.0):
    conf, correct = [], []
    for r in rows:
        p = scaled_probs(r['logp'], T)
        conf.append(p.max())
        correct.append(int(p.argmax() == r['answer']))
    return conf, correct


def nll(rows, T):
    """Average cross-entropy: -log(probability on the correct option)."""
    return float(np.mean([-np.log(scaled_probs(r['logp'], T)[r['answer']] + 1e-12) for r in rows]))


def fit_temperature(rows):
    grid = np.exp(np.linspace(np.log(0.5), np.log(20), 200))   # 0.5 .. 20, evenly spaced in log
    losses = [nll(rows, T) for T in grid]
    return float(grid[int(np.argmin(losses))])


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'Qwen3-1.7B'
    cal, test = load(model, 'cal'), load(model, 'test')
    T = fit_temperature(cal)
    print(f'{model}: fitted T = {T:.2f}  (cal NLL {nll(cal, 1.0):.3f} -> {nll(cal, T):.3f})\n')

    print(f"{'':16s} {'acc':>6} {'conf':>6} {'ECE':>6} {'Brier':>6} {'sel@50':>7}")
    for group, tasks in [('familiar', FAMILIAR), ('unfamiliar', UNFAMILIAR)]:
        rows = [r for r in test if r['task'] in tasks]
        for label, t in [('before (T=1)', 1.0), (f'after (T={T:.2f})', T)]:
            s = summary(*conf_correct(rows, t))
            print(f"{group + ' ' + label:28s} {s['acc']:6.3f} {s['avg_conf']:6.3f} {s['ece']:6.3f} {s['brier']:6.3f} {s['sel_acc@50']:7.3f}")
        print()

    print('per task ECE (before -> after):')
    for task in FAMILIAR + UNFAMILIAR:
        rows = [r for r in test if r['task'] == task]
        b, a = summary(*conf_correct(rows, 1.0)), summary(*conf_correct(rows, T))
        tag = 'familiar  ' if task in FAMILIAR else 'unfamiliar'
        print(f"  {tag} {task:14s} {b['ece']:.3f} -> {a['ece']:.3f}   (avg conf {b['avg_conf']:.2f} -> {a['avg_conf']:.2f}, acc {a['acc']:.2f})")

    print('\nreliability after scaling, unfamiliar tasks:')
    rows = [r for r in test if r['task'] in UNFAMILIAR]
    for b, n, c, a in reliability(*conf_correct(rows, T)):
        print(f'  {b}  n={n:5d}  conf {c:.2f}  acc {a:.2f}  gap {c - a:+.2f}')
