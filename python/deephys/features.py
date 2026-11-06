"""Ports of waveform timing, activity, ACG and harmonization primitives.

The complete native feature CSV can also be read without losing unported
features. Rise/Decay from MATLAB slewrate are deliberately not substituted
with a different slope definition here.
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy.integrate import trapezoid
from scipy.signal import convolve
from scipy.stats import skew, kurtosis
from .numeric import colon_edges


ACTIVITY_COLUMNS = [
    "FiringRate",
    "MeanInterSpikeInterval",
    "VarianceInterSpikeInterval",
    "CVInterSpikeInterval",
    "CV2InterSpikeInterval",
    "LocalVariation",
    "RevisedLocalVariation",
    "FanoFactor",
    "PartialAutocorrelation",
    "ISIBimodality",
]


def activity_features(
    spike_times, duration, *, fano_bin_width=0.1, refractory_period=0.005
):
    """Native computeActivityFeatures scalar definitions; SI times in seconds."""
    t = np.asarray(spike_times, float).ravel()
    if duration <= 0 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError("finite sorted spike times and positive duration required")
    out = dict.fromkeys(ACTIVITY_COLUMNS, np.nan)
    if len(t) <= 2:
        return out
    isi = np.diff(t)
    mean = isi.mean()
    out.update(
        FiringRate=len(t) / duration,
        MeanInterSpikeInterval=mean,
        VarianceInterSpikeInterval=isi.var(ddof=1),
        CVInterSpikeInterval=isi.std(ddof=1) / mean if mean > 0 else np.nan,
    )
    a, b = isi[:-1], isi[1:]
    d = a + b
    good = d > 0
    if good.any():
        out["CV2InterSpikeInterval"] = np.mean(2 * np.abs(b[good] - a[good]) / d[good])
        out["LocalVariation"] = np.mean(3 * (b[good] - a[good]) ** 2 / d[good] ** 2)
        out["RevisedLocalVariation"] = 3 * np.mean(
            (1 - 4 * a[good] * b[good] / d[good] ** 2)
            * (1 + 4 * refractory_period / d[good])
        )
    edges = colon_edges(0.0, fano_bin_width, duration)
    if len(edges) > 2:
        counts = np.histogram(t, edges)[0]
        if counts.mean() > 0:
            out["FanoFactor"] = counts.var(ddof=1) / counts.mean()
    if len(isi) > 3:
        x = isi[:-1] - isi[:-1].mean()
        y = isi[1:] - isi[1:].mean()
        if np.dot(x, x) > 0:
            out["PartialAutocorrelation"] = np.dot(x, y) / np.dot(x, x)
    log_isi = np.log(isi[isi > 0])
    if len(log_isi) >= 10:
        k = kurtosis(log_isi, fisher=False, bias=True)
        if k >= 0.1:
            out["ISIBimodality"] = (skew(log_isi, bias=True) ** 2 + 1) / k
    return out


def autocorrelogram(
    spike_times, bin_size=0.0005, lag=0.1, quantization_period=1 / 20000
):
    """FMAToolbox centered bins, excluding each spike's self-pair.

    Counts are ordered pairs. Native CCG first rounds timestamps to its clock
    (default 50 us). Its floor(0.5 + signed_lag/bin_size) tie convention can
    make negative and positive half-bin ties asymmetric.
    """
    t = np.asarray(spike_times, float).ravel()
    if not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        raise ValueError("spikes must be finite and sorted")
    half = int(np.floor(lag / bin_size + 0.5))
    clock = np.floor(t / quantization_period + 0.5).astype(np.int64)
    width = int(np.floor(bin_size / quantization_period + 0.5))
    if width < 1:
        raise ValueError("CCG bin size is smaller than its quantization clock")
    edge = (half + 0.5) * width
    counts = np.zeros(2 * half + 1, np.int64)
    for i, x in enumerate(clock):
        end = np.searchsorted(clock, x + edge, side="right")
        delta = clock[i + 1 : end] - x
        pos = delta[delta < edge]
        pbin = half + np.floor(0.5 + pos / width).astype(int)
        nbin = half + np.floor(0.5 - delta / width).astype(int)
        counts += np.bincount(pbin, minlength=len(counts))
        counts += np.bincount(nbin, minlength=len(counts))
    # The MATLAB wrapper ends with flipud(counts), including tie asymmetry.
    return counts[::-1]


def waveform_shape(waveform, sampling_rate):
    """Seven directly portable native features on a 10x cubic-spline grid.

    Width and T2Pdelay are in ms; normalized-waveform AUC is amplitude x ms.
    MATLAB state-level/slewrate Rise and Decay are outside this function.
    """
    wf = np.asarray(waveform, float).ravel()
    names = (
        "AUC_peak_1",
        "AUC_trough",
        "AUC_peak_2",
        "HalfWidth",
        "Asymmetry",
        "T2Pdelay",
        "T2Pratio",
    )
    if len(wf) < 3 or not np.isfinite(wf).all() or sampling_rate <= 0:
        return dict.fromkeys(names, np.nan)
    xq = np.arange(10 * (len(wf) - 1) + 1) / 10
    w = CubicSpline(np.arange(len(wf)), wf)(xq)
    ti = int(w.argmin())
    per_ms = int(np.floor(sampling_rate / 1000 * 10 + 0.5))
    p1 = w[max(0, ti - per_ms) : ti + 1].max()
    p2i = ti + int(w[ti : min(len(w), ti + 2 * per_ms + 1)].argmax())
    p2 = w[p2i]
    den = p1 + p2
    asym = (
        (p2 - p1) / den
        if max(abs(p1), abs(p2)) >= 1e-9 and abs(den) >= 1e-6 * max(abs(p1), abs(p2))
        else np.nan
    )
    low = w < w[ti] / 2
    if low[ti]:
        before = np.flatnonzero(~low[: ti + 1])
        after = np.flatnonzero(~low[ti:])
        h0 = int(before[-1] + 1) if len(before) else 0
        h1 = int(ti + after[0] - 1) if len(after) else len(w) - 1
        width = (h1 - h0) / (sampling_rate * 10 / 1000)
    else:
        width = np.nan
    # Retain the author's inclusive endpoint/index conventions in the AUCs.
    zc = np.r_[1, np.flatnonzero(w[:-1] * w[1:] <= 0) + 1, len(w)].astype(int)
    prior = np.flatnonzero(zc < ti + 1)
    zi = max(int(prior[-1]) if len(prior) else 0, 1)
    trap = lambda a, b: trapezoid(w[a - 1 : b]) / per_ms
    auc1 = trap(zc[zi - 1], zc[zi])
    auct = abs(trap(zc[zi], zc[zi + 1])) if zi + 1 < len(zc) else np.nan
    if zi + 1 < len(zc):
        auc2 = trap(zc[zi + 1], zc[zi + 2] if len(zc[zi:]) > 3 else len(w))
    else:
        auc2 = np.nan
    return dict(
        AUC_peak_1=auc1,
        AUC_trough=auct,
        AUC_peak_2=auc2,
        HalfWidth=width,
        Asymmetry=asym,
        T2Pdelay=(p2i - ti) / (sampling_rate * 10 / 1000),
        T2Pratio=abs(w[ti] / p2) if p2 != 0 else np.nan,
    )


@dataclass(frozen=True)
class Harmonization:
    target_rate: float = 120000.0
    pre_ms: float = 1.0
    post_ms: float = 2.0
    edge_mode: str = "edge"
    acg_normalization: str = "peak"
    acg_bin_size: float = 0.0005
    acg_lag: float = 0.1


def harmonize(waveforms, acgs, sampling_rate, config=None):
    """Native buildFeatureMatrix: samples x units -> units x features.

    Uses modified Akima interpolation and keeps ACG columns before waveforms.
    """
    p = config or Harmonization()
    wf, acg = np.asarray(waveforms, float), np.asarray(acgs, float)
    if wf.ndim == 1:
        wf = wf[:, None]
    if acg.ndim == 1:
        acg = acg[:, None]
    if wf.shape[1] != acg.shape[1]:
        raise ValueError("unit counts differ")
    expected = int(np.floor(2 * p.acg_lag / p.acg_bin_size + 0.5)) + 1
    if acg.shape[0] != expected:
        raise ValueError("ACG parameters do not match bin count")
    sr = np.broadcast_to(np.asarray(sampling_rate, float), (wf.shape[1],))
    pre, post = p.pre_ms, p.post_ms
    cleaned = []
    for i in range(wf.shape[1]):
        w = wf[:, i]
        finite = np.flatnonzero(np.isfinite(w))
        w = w[finite[0] : finite[-1] + 1] if len(finite) else np.array([])
        if not np.isfinite(w).all():
            raise ValueError("internal NaN waveform samples")
        cleaned.append(w)
        if p.edge_mode == "trim" and len(w) and np.any(w):
            ti = w.argmin()
            pre = min(pre, ti * 1000 / sr[i])
            post = min(post, (len(w) - ti - 1) * 1000 / sr[i])
    npre = int(np.floor(pre * p.target_rate / 1000 + 0.5))
    npost = int(np.floor(post * p.target_rate / 1000 + 0.5))
    grid = np.arange(-npre, npost + 1) * 1000 / p.target_rate
    aligned = np.full((len(grid), wf.shape[1]), np.nan)
    for i, w in enumerate(cleaned):
        if len(w) < 3 or not np.any(w):
            continue
        t = (np.arange(len(w)) - w.argmin()) * 1000 / sr[i]
        inside = (grid >= t[0]) & (grid <= t[-1])
        if inside.sum() < 3:
            continue
        value = np.zeros(len(grid))
        value[inside] = Akima1DInterpolator(t, w, method="makima")(grid[inside])
        if p.edge_mode == "edge":
            where = np.flatnonzero(inside)
            value[: where[0]] = value[where[0]]
            value[where[-1] + 1 :] = value[where[-1]]
        elif p.edge_mode not in ("zero", "trim"):
            raise ValueError("unknown edge mode")
        amp = np.max(np.abs(value))
        if amp > 0:
            aligned[:, i] = value / amp
    if p.acg_normalization == "baseline":
        nt = max(1, int(np.floor(0.2 * (acg.shape[0] - 1) / 2 + 0.5)))
        denom = np.mean(np.r_[acg[:nt], acg[-nt:]], axis=0)
    elif p.acg_normalization == "peak":
        denom = np.max(acg, axis=0)
    else:
        raise ValueError("unknown ACG normalization")
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = acg / denom
    norm[~np.isfinite(norm)] = 0
    X = np.c_[norm.T, aligned.T]
    names = [f"FullACG{i + 1}" for i in range(len(norm))] + [
        f"Waveform{i + 1}" for i in range(len(aligned))
    ]
    return X, names, aligned, norm


def smooth_acg(acgs, sigma_bins):
    """Native Gaussian smoothing with excluded/preserved center and edge renormalization."""
    acg = np.asarray(acgs, float)
    if sigma_bins <= 0 or acg.size == 0:
        return acg.copy()
    one_dim = acg.ndim == 1
    if one_dim:
        acg = acg[:, None]
    center = (len(acg) - 1) // 2
    h = int(np.ceil(4 * sigma_bins))
    kernel = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma_bins) ** 2)
    kernel /= kernel.sum()
    x = acg.copy()
    x[center] = 0
    weights = np.ones((len(acg), 1))
    weights[center] = 0
    numerator = convolve(x, kernel[:, None], mode="same", method="direct")
    denominator = convolve(weights, kernel[:, None], mode="same", method="direct")
    result = numerator / np.maximum(denominator, np.finfo(float).eps)
    result[center] = acg[center]
    return result[:, 0] if one_dim else result


def classifier_features(waveforms, acgs, sampling_rate, *, smoothing_ms=0.0):
    """Single-recording native preprocessing and retained feature names."""
    from .graphs import global_normalize

    raw, names, aligned, normalized = harmonize(
        waveforms, smooth_acg(acgs, smoothing_ms / 0.5), sampling_rate
    )
    missing = np.isnan(raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        grouped = (raw - np.nanmean(raw, axis=0)) / np.nanstd(raw, axis=0, ddof=1)
    grouped[missing] = 0
    X, parameters = global_normalize(grouped)
    return X, np.asarray(names)[parameters["keep"]], aligned, normalized
