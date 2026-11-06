"""Population E/I analysis using explicit, externally supported unit labels.

Labels: 0 unknown, 1 excitatory, 2 inhibitory. No cell-type prior is inferred
from genotype, community number, or firing rate. Activity is stored in
spikes/bin; use the *_hz properties for rates in Hz.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from .numeric import bin_spikes, movmean, percentile


@dataclass
class Activity:
    E: np.ndarray
    I: np.ndarray
    total: np.ndarray
    ratio: np.ndarray
    E_raw: np.ndarray
    I_raw: np.ndarray
    binned_mat: np.ndarray
    labels: np.ndarray
    x: np.ndarray
    bin_size: float
    rate_type: str

    @property
    def E_hz(self):
        return self.E / self.bin_size

    @property
    def I_hz(self):
        return self.I / self.bin_size

    @property
    def total_hz(self):
        return self.total / self.bin_size


def activity_from_counts(
    counts,
    labels,
    bin_size=0.01,
    smooth_window=5,
    rate_type="mean",
    ratio_method="smoothed",
    start=0.0,
):
    mat = np.asarray(counts, float)
    labels = np.asarray(labels)
    if mat.ndim != 2 or len(labels) != len(mat) or np.any(~np.isin(labels, [0, 1, 2])):
        raise ValueError("one label 0/1/2 per unit is required")
    if rate_type not in ("mean", "sum"):
        raise ValueError("rate_type must be mean or sum")
    reduce = np.mean if rate_type == "mean" else np.sum

    def pop(mask):
        return reduce(mat[mask], axis=0) if np.any(mask) else np.zeros(mat.shape[1])

    e_raw = pop(labels == 1)
    i_raw = pop(labels == 2)
    total_raw = pop(np.isin(labels, [1, 2]))
    e = movmean(e_raw, smooth_window)
    i = movmean(i_raw, smooth_window)
    total = movmean(total_raw, smooth_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        if ratio_method == "perbin":
            ratio = np.where(e_raw > 0, i_raw / e_raw, np.nan)
            ratio = movmean(ratio, smooth_window, omitnan=True)
        elif ratio_method == "smoothed":
            ratio = np.where(e > 0, i / e, np.nan)
        else:
            raise ValueError("unknown ratio method")
    return Activity(
        e,
        i,
        total,
        ratio,
        e_raw,
        i_raw,
        mat,
        labels,
        start + np.arange(mat.shape[1]) * bin_size,
        bin_size,
        rate_type,
    )


def compute_activity(
    trains,
    labels,
    duration,
    *,
    bin_size=0.01,
    smooth_window=5,
    rate_type="mean",
    ratio_method="smoothed",
    start=0.0,
):
    counts, _ = bin_spikes(trains, duration, bin_size, start=start)
    return activity_from_counts(
        counts, labels, bin_size, smooth_window, rate_type, ratio_method, start
    )


def detect_ei_states(
    activity, *, max_ie_ratio=2.0, noise_floor_percentile=5.0, smooth_window=9
):
    """Port of EIAnalyzer.detectBursts (E-dominant state, not ISINJoint)."""
    if smooth_window % 2 != 1:
        raise ValueError("odd median filter window required")
    pos = activity.E[activity.E > 0]
    floor = percentile(pos, noise_floor_percentile) if len(pos) else 0.0
    state = (activity.ratio < max_ie_ratio) & (activity.E > floor)
    return median_filter(
        state.astype(float), size=smooth_window, mode="constant", cval=0
    ).astype(bool), float(floor)


def population_correlations(activity):
    """Leave-one-out self-class population correlations; columns [I,E]."""
    mat = activity.binned_mat
    labels = activity.labels
    result = np.full((len(mat), 2), np.nan)
    for j, label in enumerate([2, 1]):
        members = labels == label
        n = int(members.sum())
        total = mat[members].sum(axis=0) if n else np.zeros(mat.shape[1])
        for i, row in enumerate(mat):
            count = n - int(members[i])
            if count <= 0:
                continue
            pop = (total - row if members[i] else total) / count
            if np.std(row) > 0 and np.std(pop) > 0:
                result[i, j] = np.corrcoef(row, pop)[0, 1]
    return result


def burst_cutouts(
    activity,
    *,
    half_window_bins=150,
    min_height=0.1,
    min_prominence=0.1,
    min_distance_bins=6,
    state=None,
):
    """Extract all qualifying population peaks, without hidden boundary padding.

    The author's later iterative template filtering/alignment is not applied
    by this primitive. That distinction is included in the coverage manifest.
    """
    trace = activity.total.copy()
    if state is not None:
        trace *= np.asarray(state, bool)
    maximum = max(float(trace.max(initial=0)), np.finfo(float).eps)
    peaks, _ = find_peaks(
        trace / maximum,
        height=min_height,
        prominence=min_prominence,
        distance=min_distance_bins,
    )
    w = int(half_window_bins)
    peaks = peaks[(peaks >= w) & (peaks < len(trace) - w)]
    offsets = np.arange(-w, w + 1)
    index = peaks[:, None] + offsets
    return dict(
        peaks=peaks,
        relative_seconds=offsets * activity.bin_size,
        total=activity.total[index],
        E=activity.E[index],
        I=activity.I[index],
        ratio=activity.ratio[index],
        method="all qualifying peaks; no native iterative acceptance/alignment",
    )


def rate_matched_null(counts, labels, *, n_draws=200, seed=42):
    """Supplemental rate-stratified label-permutation null.

    Class counts are preserved within firing-count strata. This is not a
    replacement for the native EIAnalyzer null or experimental validation.
    """
    mat = np.asarray(counts, float)
    y = np.asarray(labels)
    known = np.isin(y, [1, 2])
    idx = np.flatnonzero(known)
    result = np.full(n_draws, np.nan)
    if not np.any(y == 1) or not np.any(y == 2):
        return result
    order = idx[np.argsort(mat[idx].sum(axis=1), kind="stable")]
    strata = np.array_split(order, max(1, min(10, len(order) // 10)))
    rng = np.random.default_rng(seed)
    for k in range(n_draws):
        shuffled = y.copy()
        for s in strata:
            shuffled[s] = rng.permutation(y[s])
        E = mat[shuffled == 1].mean(axis=0)
        I = mat[shuffled == 2].mean(axis=0)
        if E.std() > 0 and I.std() > 0:
            result[k] = np.corrcoef(E, I)[0, 1]
    return result
