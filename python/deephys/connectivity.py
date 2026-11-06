"""Spike timing tiling coefficient: Cutts & Eglen (2014).

The scalar arithmetic follows the bundled DeePhys MATLAB translation,
including its undefined-denominator NaNs. No fastmath is used.
"""

import numpy as np
from scipy.stats import norm
from .numeric import bh_fdr


def _tile(t, dt, duration):
    n = len(t)
    if not n:
        return np.nan
    value = 2 * n * dt
    if n == 1:
        if t[0] < dt:
            value = value + t[0] - dt
        elif t[0] + dt > duration:
            value = value - t[0] - dt + duration
    else:
        for i in range(n - 1):
            gap = t[i + 1] - t[i]
            if gap < 2 * dt:
                value = value - 2 * dt + gap
        if t[0] < dt:
            value = value + t[0] - dt
        if duration - t[-1] < dt:
            value = value - t[-1] - dt + duration
    return value / duration


def _coincident(a, b, dt):
    count = 0
    j = 0
    for x in a:
        while j < len(b):
            if abs(x - b[j]) <= dt:
                count += 1
                break
            if b[j] > x:
                break
            j += 1
    return count / len(a)


def sttc(a, b, *, duration, dt=0.005):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if duration <= 0 or dt <= 0:
        raise ValueError("positive duration/dt required")
    if any(
        np.any(np.diff(t) < 0)
        or not np.isfinite(t).all()
        or np.any((t < 0) | (t > duration))
        for t in [a, b]
    ):
        raise ValueError("finite sorted in-window spikes required")
    if not len(a) or not len(b):
        return np.nan
    ta, tb = _tile(a, dt, duration), _tile(b, dt, duration)
    pa, pb = _coincident(a, b, dt), _coincident(b, a, dt)
    d1, d2 = 1 - tb * pa, 1 - ta * pb
    if d1 == 0 or d2 == 0:
        return np.nan
    return 0.5 * (pa - tb) / d1 + 0.5 * (pb - ta) / d2


def sttc_matrix(trains, *, duration, dt=0.005):
    n = len(trains)
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            result[i, j] = result[j, i] = sttc(
                trains[i], trains[j], duration=duration, dt=dt
            )
    return result


def sttc_network(
    trains, *, duration, dt=0.005, jitter=0.05, n_surrogates=100, alpha=0.05, seed=42
):
    """Streaming jitter null with Gaussian approximation and BH-FDR.

    NumPy and MATLAB Gaussian random streams differ. Empirical coefficients
    have direct parity tests; randomized p/q values need identical supplied
    surrogate draws before a bitwise-parity claim would be valid.
    """
    if n_surrogates < 2:
        raise ValueError("at least 2 null draws required")
    empirical = sttc_matrix(trains, duration=duration, dt=dt)
    mean = np.zeros_like(empirical)
    m2 = np.zeros_like(empirical)
    rng = np.random.default_rng(seed)
    for k in range(1, n_surrogates + 1):
        shifted = [
            np.sort(np.mod(t + rng.normal(0, jitter, len(t)), duration)) for t in trains
        ]
        sample = sttc_matrix(shifted, duration=duration, dt=dt)
        delta = sample - mean
        mean += delta / k
        m2 += delta * (sample - mean)
    sd = np.sqrt(m2 / (n_surrogates - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (empirical - mean) / sd
    p = norm.sf(z)
    ii, jj = np.triu_indices(len(trains), 1)
    q = np.ones_like(p)
    q[ii, jj] = bh_fdr(p[ii, jj])
    q[jj, ii] = q[ii, jj]
    edges = (q < alpha) & (empirical > 0)
    np.fill_diagonal(edges, False)
    return dict(weights=empirical, p=p, q=q, edges=edges, null_mean=mean, null_sd=sd)
