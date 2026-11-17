"""One table comparing every method on the test set (familiar vs unfamiliar).

Spread methods get one global temperature fit on familiar calibration data (calibrate.py).
Written methods use the number the model wrote (format failures = wrong at confidence 1.0).

Run: python report.py
"""
import json
import os

import numpy as np

from calibrate import conf_correct, fit_temperature, load
from data import FAMILIAR, UNFAMILIAR
from metrics import summary

SPREAD = [('base + T', 'Qwen3-1.7B'), ('fine-tuned + T', 'sft')]
WRITTEN = [('base, written', 'Qwen3-1.7B'), ('RL (Brier), written', 'rlw_brier')]


def row(name, conf, correct, fmt=None):
    s = summary(conf, correct)
    return f"{name:22s} {s['acc']:6.3f} {s['ece']:6.3f} {s['brier']:6.3f} {s['sel_acc@50']:7.3f}" + (f' {fmt:6.1%}' if fmt is not None else '')


if __name__ == '__main__':
    for group, tasks in [('FAMILIAR', FAMILIAR), ('UNFAMILIAR', UNFAMILIAR)]:
        print(f"\n{group:22s} {'acc':>6} {'ECE':>6} {'Brier':>6} {'sel@50':>7} {'fmt fail':>8}")
        for name, tag in SPREAD:
            if not os.path.exists(f'results/spread_{tag}_test.jsonl'):
                continue
            T = fit_temperature(load(tag, 'cal'))
            rows = [r for r in load(tag, 'test') if r['task'] in tasks]
            print(row(f'{name} (T={T:.2f})', *conf_correct(rows, T)))
        for name, tag in WRITTEN:
            path = f'results/written_{tag}_test.jsonl'
            if not os.path.exists(path):
                continue
            rows = [json.loads(l) for l in open(path) if json.loads(l)['task'] in tasks]
            print(row(name, [r['conf'] for r in rows], [r['correct'] for r in rows],
                      1 - np.mean([r['format_ok'] for r in rows])))
