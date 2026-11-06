"""Small MATLAB-compatible numerical conventions used by the companion."""

import numpy as np


def colon_edges(start, step, stop):
    """Symmetric-endpoint MATLAB colon construction for increasing uniform edges.

    Constructing every edge as i*step changes bins for spikes exactly on the
    recording sample grid. The right half is evaluated from the endpoint.
    """
    if step <= 0 or stop < start:
        raise ValueError("increasing positive-step range required")
    n = int(np.floor((stop - start) / step + 1e-10))
    endpoint = start + n * step
    if abs(endpoint - stop) <= 2 * np.finfo(float).eps * max(abs(start), abs(stop), 1):
        endpoint = stop
    values = start + np.arange(n + 1) * step
    middle = n // 2
    values[middle + 1 :] = endpoint - np.arange(n - middle - 1, -1, -1) * step
    return values


def percentile(values, q, axis=None):
    """MATLAB prctile convention (Hazen plotting positions), omitting NaNs."""
    return np.nanpercentile(values, q, axis=axis, method="hazen")


def movmean(values, window, *, omitnan=False, axis=-1):
    """MATLAB movmean with shrinking endpoints; even windows look backwards."""
    a = np.moveaxis(np.asarray(values, float), axis, -1)
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    n = a.shape[-1]
    left, right = window // 2, (window - 1) // 2
    lo = np.maximum(np.arange(n) - left, 0)
    hi = np.minimum(np.arange(n) + right + 1, n)
    good = np.isfinite(a)
    cs = np.concatenate(
        [np.zeros(a.shape[:-1] + (1,)), np.cumsum(np.where(good, a, 0), axis=-1)],
        axis=-1,
    )
    cn = np.concatenate(
        [np.zeros(a.shape[:-1] + (1,)), np.cumsum(good, axis=-1)], axis=-1
    )
    counts = cn[..., hi] - cn[..., lo]
    sums = cs[..., hi] - cs[..., lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = sums / counts
    if not omitnan:
        result = np.where(counts == hi - lo, result, np.nan)
    return np.moveaxis(result, -1, axis)


def bin_spikes(trains, duration, bin_size=0.01, *, start=0.0):
    """Closed-left, open-right window; a final incomplete bin is omitted."""
    if not (np.isfinite(duration) and duration > start and bin_size > 0):
        raise ValueError("invalid time window or bin size")
    nb = int(np.floor((duration - start) / bin_size + 1e-10))
    edges = colon_edges(start, bin_size, duration)
    result = np.zeros((len(trains), nb), dtype=np.int64)
    for i, train in enumerate(trains):
        t = np.asarray(train, float)
        t = t[(t >= start) & (t < duration)]
        result[i] = np.histogram(t, edges)[0]
    return result, edges


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    v = p[ok]
    order = np.argsort(v, kind="stable")
    adjusted = np.minimum.accumulate(
        (v[order] * len(v) / np.arange(1, len(v) + 1))[::-1]
    )[::-1]
    restored = np.empty(len(v))
    restored[order] = adjusted.clip(0, 1)
    q[ok] = restored
    return q
