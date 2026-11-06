"""Numerical tests against synthetic cases executed in MATLAB R2023b.

No research spikes, labels, recordings or identifiable metadata are included.
"""

from pathlib import Path
import json
import numpy as np
import pytest
from deephys.features import (
    activity_features,
    autocorrelogram,
    waveform_shape,
    harmonize,
    smooth_acg,
)
from deephys.graphs import fuzzy_knn_graph, propagate_labels, global_normalize
from deephys.ei import activity_from_counts, detect_ei_states, population_correlations
from deephys.connectivity import sttc_matrix

GOLD = json.loads(
    (Path(__file__).parent / "fixtures" / "matlab_synthetic_reference.json").read_text()
)


@pytest.mark.parametrize("unit", range(5))
def test_activity_scalars(unit):
    got = activity_features(GOLD["times"][unit], GOLD["duration"])
    ref = GOLD["activity"][unit]
    for key, val in ref.items():
        np.testing.assert_allclose(
            got[key],
            np.nan if val is None else val,
            rtol=1e-10,
            atol=1e-12,
            err_msg=key,
        )


@pytest.mark.parametrize("unit", range(5))
def test_centered_acg_counts(unit):
    got = autocorrelogram(GOLD["times"][unit])
    np.testing.assert_array_equal(got, np.asarray(GOLD["acgs"])[:, unit])


@pytest.mark.parametrize("unit", range(5))
def test_waveform_shape(unit):
    got = waveform_shape(np.asarray(GOLD["waveforms"])[:, unit], GOLD["sampling_rate"])
    ref = GOLD["shape"][unit]
    for key, value in got.items():
        expected = np.nan if ref[key] is None else ref[key]
        np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-12, err_msg=key)


def test_makima_harmonization():
    X, names, w, a = harmonize(GOLD["waveforms"], GOLD["acgs"], GOLD["sampling_rate"])
    # MATLAB/SciPy makima flatten almost-zero derivative weights differently.
    # Max measured discrepancy on these normalized synthetic waveforms: 6.7e-11.
    np.testing.assert_allclose(X, GOLD["harmonized_features"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(w, GOLD["aligned_waveforms"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a, GOLD["normalized_acgs"], rtol=1e-12, atol=1e-12)
    assert names == GOLD["feature_names"]


def test_exact_fuzzy_graph():
    W = fuzzy_knn_graph(GOLD["graph_X"], 3)
    np.testing.assert_allclose(W.toarray(), GOLD["graph"], rtol=1e-10, atol=1e-12)


def test_clamped_propagation_and_seed_count_normalization():
    got = propagate_labels(
        GOLD["graph"], GOLD["seed_indices_zero_based"], GOLD["seed_labels"], 1000, 1e-9
    )
    ref = GOLD["propagation"]
    np.testing.assert_allclose(got.scores, ref["scores"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(got.unreached, np.asarray(ref["unreached"]).ravel())
    assert got.converged == ref["converged"]
    assert got.iterations == ref["iterations"]


@pytest.mark.parametrize("ratio_method", ["smoothed", "perbin"])
def test_ei_activity_and_ratio(ratio_method):
    ref = GOLD["ei"]
    got = activity_from_counts(
        ref["counts"],
        ref["labels"],
        ref["bin_size"],
        ref["smooth_window"],
        ratio_method=ratio_method,
    )
    for key in ["E", "I", "total", "E_raw", "I_raw"]:
        np.testing.assert_allclose(getattr(got, key), ref[key], rtol=1e-12, atol=1e-12)
    target = "ratio" if ratio_method == "smoothed" else "perbin_ratio"
    expected = np.array([np.nan if x is None else x for x in ref[target]])
    np.testing.assert_allclose(got.ratio, expected, rtol=1e-12, atol=1e-12)


def test_e_dominant_burst_state():
    ref = GOLD["ei"]
    a = activity_from_counts(ref["counts"], ref["labels"])
    state, _ = detect_ei_states(a)
    np.testing.assert_array_equal(state, ref["burst_state"])


def test_leave_one_out_correlations():
    ref = GOLD["ei"]
    a = activity_from_counts(ref["counts"], ref["labels"])
    np.testing.assert_allclose(
        population_correlations(a),
        np.array(ref["population_correlations"], float),
        rtol=1e-11,
        atol=1e-12,
    )


def test_native_sttc_coefficients():
    got = sttc_matrix([np.array(t) for t in GOLD["times"]], duration=GOLD["duration"])
    np.testing.assert_allclose(
        got, np.array(GOLD["sttc"], float), rtol=1e-12, atol=1e-12
    )


def test_center_preserving_acg_smoothing():
    got = smooth_acg(GOLD["acgs"], 3.5 / 0.5)
    np.testing.assert_allclose(got, GOLD["smoothed_acgs"], rtol=1e-12, atol=1e-12)


def test_native_global_normalization_constant_columns():
    got, params = global_normalize(np.array(GOLD["normalization_input"], float))
    np.testing.assert_array_equal(params["keep"], GOLD["normalization_keep"])
    np.testing.assert_allclose(
        got, GOLD["normalization_expected"], rtol=1e-12, atol=1e-12
    )
