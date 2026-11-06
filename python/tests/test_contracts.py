import sys, os
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from deephys.numeric import movmean, bin_spikes, bh_fdr
from deephys.graphs import fuzzy_knn_graph, propagate_labels
from deephys.phy import PhyRecording, read_params
from deephys.ei import activity_from_counts, population_correlations
from deephys.phenotype import grouped_prediction
from deephys.safety import install_readonly_guard
from deephys.store import FeatureStore


def test_closed_open_recording_window():
    counts, edges = bin_spikes([np.array([0.0, 0.1, 0.999, 1.0])], 1.0, 0.1)
    assert counts.sum() == 3
    assert counts[0, 0] == counts[0, 1] == counts[0, -1] == 1


def test_even_movmean_and_shrinking_endpoints():
    np.testing.assert_allclose(movmean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(movmean([1, np.nan, 3], 3, omitnan=True), [1, 2, 3])


def test_bh_fdr_preserves_nan_locations():
    np.testing.assert_allclose(
        bh_fdr([0.01, 0.04, 0.03, np.nan]), [0.03, 0.04, 0.04, np.nan]
    )


def test_disconnected_units_stay_unknown():
    G = sparse.block_diag(
        [np.ones((3, 3)) - np.eye(3), np.ones((2, 2)) - np.eye(2), np.zeros((1, 1))]
    )
    p = propagate_labels(G, [0, 1], [1, 2])
    assert p.unreached[3:].all()


def test_duplicate_feature_rows_give_finite_symmetric_graph():
    X = np.array([[0, 0], [0, 0], [1, 1], [2, 2]], float)
    W = fuzzy_knn_graph(X, 2).toarray()
    assert np.isfinite(W).all()
    np.testing.assert_allclose(W, W.T)
    assert not W.diagonal().any()


def test_params_are_never_executed(tmp_path):
    marker = tmp_path / "must_not_exist"
    p = tmp_path / "params.py"
    p.write_text(
        f"sample_rate = 10000.\nn_samples = 20\nopen({str(marker)!r}, 'w').write('bad')\n"
    )
    assert read_params(p)["sample_rate"] == 10000
    assert not marker.exists()


def test_original_phy_ids_and_explicit_calibration(tmp_path):
    (tmp_path / "params.py").write_text("sample_rate=1000.\nn_samples=10\n")
    np.save(tmp_path / "spike_times.npy", np.array([1, 1, 4, 9], dtype=np.int64))
    np.save(tmp_path / "spike_clusters.npy", np.array([100, 100, 101, 101]))
    np.save(tmp_path / "templates.npy", np.arange(20, dtype=float).reshape(2, 5, 2))
    np.save(tmp_path / "phy_ids.npy", np.array([101, 100]))
    np.save(tmp_path / "channel_positions.npy", np.array([[0, 0], [10, 0]]))
    before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    p = PhyRecording.read(tmp_path)
    assert p.duplicate_counts() == {100: 1, 101: 0}
    with pytest.raises(ValueError, match="calibration"):
        p.physical_template(100)
    p = PhyRecording.read(tmp_path, template_gain_uv=2.0)
    np.testing.assert_array_equal(
        p.physical_template(100), np.arange(10, 20).reshape(5, 2) * 2
    )
    assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


def test_unknown_units_do_not_contribute_to_classified_population():
    counts = np.array([[1, 0, 2], [2, 1, 0], [100, 100, 100]])
    a = activity_from_counts(counts, [1, 2, 0], smooth_window=1)
    np.testing.assert_allclose(a.total, [1.5, 0.5, 1.0])
    assert np.isnan(a.ratio[1])
    # Conservative departure from native self-correlation when only one class member exists.
    assert np.isnan(population_correlations(a)[0, 1])


def test_grouped_model_keeps_test_group_out_and_fits_imputer_on_train_only():
    X = pd.DataFrame(
        {
            "signal": [0, 1, 0, 1, 0, 1, 0, 1],
            "missing": [np.nan] * 4 + [1000, 1000, 1000, 1000],
        }
    )
    y = np.array(["A", "B"] * 4)
    groups = np.array(["N1"] * 4 + ["N3"] * 4)
    pred, metrics, importance, models = grouped_prediction(
        X, y, groups, trees=5, n_jobs=1
    )
    assert len(pred) == 8 and len(metrics) == 2
    assert set(pred.TestGroup) == {"N1", "N3"}
    assert models[1].named_steps["imputer"].statistics_[1] == 0


def test_protected_input_and_network_write_events_are_denied_without_os_write(tmp_path):
    protect = tmp_path / "not_created"
    install_readonly_guard([protect])
    for p in [str(protect / "no_file"), r"\\nas\share\no_file"]:
        with pytest.raises(PermissionError):
            sys.audit("open", p, "w", os.O_WRONLY | os.O_CREAT)
    assert not protect.exists()


def test_predictor_contract_excludes_history_identifiers_and_processing_stage():
    data = pd.DataFrame(
        {
            "FiringRate": [1.0, 2.0],
            "HalfWidth": [0.2, 0.3],
            "OriginalUnitID": [100, 101],
            "Parent_ACG1": [3.0, 4.0],
            "ReferenceElectrode": [5, 6],
            "X_um": [100.0, 200.0],
            "MeanFC_NearNeighbors": [np.nan, 0.2],
            "SpatialDecayTau_um": [np.nan, 20.0],
            "Degree_STTC": [np.nan, 4.0],
            "DIV": [15, 18],
            "Well": [1, 2],
        }
    )
    store = FeatureStore(data, pd.DataFrame(), {})
    X, md = store.matrix()
    assert list(X) == ["FiringRate", "HalfWidth"]
    assert "OriginalUnitID" in md
