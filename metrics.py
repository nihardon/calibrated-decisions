"""Calibration metrics. All take per-question arrays of
    conf    — confidence in the chosen answer (0..1)
    correct — 1 if the chosen answer was right, else 0
so the same metrics apply to both methods (read-the-spread and written confidence).
"""
import numpy as np


def ece(conf, correct, n_bins=10):
    """Expected calibration error: bucket by confidence, weighted avg of |accuracy - confidence|."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return total


def brier(conf, correct):
    """Top-label Brier: mean of (confidence - correct)^2. 0 is perfect."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2))


def selective_accuracy(conf, correct, coverage=0.5):
    """Accuracy on the `coverage` fraction of questions the model is most confident about."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    k = max(1, int(round(len(conf) * coverage)))
    top = np.argsort(-conf, kind='stable')[:k]
    return float(correct[top].mean())


def reliability(conf, correct, n_bins=10):
    """Rows of (bucket range, count, avg confidence, accuracy) for a reliability diagram."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append((f'{b/n_bins:.1f}-{(b+1)/n_bins:.1f}', int(mask.sum()), conf[mask].mean(), correct[mask].mean()))
    return rows


def summary(conf, correct):
    return {'n': len(conf), 'acc': float(np.mean(correct)), 'avg_conf': float(np.mean(conf)),
            'ece': ece(conf, correct), 'brier': brier(conf, correct), 'sel_acc@50': selective_accuracy(conf, correct)}
