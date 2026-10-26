"""NumPy-only physical nearest-state diagnostics, independent of training runtime."""
import numpy as np


def phase_match(actual, expert, weights, search_radius=None):
    """Weighted physical RMS distance; matcher indices are diagnostic only."""
    n = len(actual['body_q']); matched = []
    for i in range(n):
        lo = 0 if search_radius is None else max(0, i-search_radius)
        hi = len(expert['body_q']) if search_radius is None else min(len(expert['body_q']), i+search_radius+1)
        d = np.zeros(hi-lo)
        for name, scale in weights.items():
            delta = expert[name][lo:hi]-actual[name][i]
            d += np.mean(delta**2, axis=-1)/(scale**2)
        matched.append(lo+int(d.argmin()))
    return np.asarray(matched)


def quantiles(values):
    v = np.asarray(values)
    return {k: float(np.percentile(v, p)) for k, p in [('p05',5),('p25',25),('p50',50),('p75',75),('p95',95)]} | {'mean': float(v.mean()), 'min':float(v.min()),'max':float(v.max())}
