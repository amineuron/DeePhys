import numpy as np
import pandas as pd
import pytest
from deephys.celltypes import classify_with_seeds, waveform_mixtures
from deephys.numeric import colon_edges


def test_seeded_classifier_leaves_disconnected_units_unknown():
    X = np.array([[0.0, 0], [0.01, 0], [0.99, 1.0], [1.0, 1.0], [10, 10], [10.1, 10.0]])
    ids = ["E_seed", "E_like", "I_like", "I_seed", "unknown_1", "unknown_2"]
    seeds = pd.DataFrame(
        {
            "UnitID": ["E_seed", "I_seed"],
            "Label": [1, 2],
            "Provenance": ["synthetic independent E", "synthetic independent I"],
        }
    )
    labels, _, _ = classify_with_seeds(X, ids, seeds, neighbors=1)
    assert labels.Label.iloc[0] == 1 and labels.Label.iloc[3] == 2
    assert labels.Label.iloc[-2:].eq(0).all()


def test_one_seed_class_is_insufficient():
    seed = pd.DataFrame({"UnitID": ["a"], "Label": [1], "Provenance": ["independent"]})
    with pytest.raises(ValueError, match="both independently"):
        classify_with_seeds(np.array([[0], [1]]), ["a", "b"], seed)


def test_no_mixture_cluster_is_called_excitatory_or_inhibitory():
    x = np.arange(60)
    frame = pd.DataFrame(
        {"HalfWidth": 0.2 + 0.01 * np.sin(x), "T2Pdelay": 0.3 + 0.03 * np.cos(x)}
    )
    labels, bic, info = waveform_mixtures(frame)
    assert set(bic.Components) == {1, 2, 3}
    assert "Label" not in labels and "E" not in labels and "I" not in labels
    assert info["selected_components"] in [1, 2, 3]


def test_uniform_colon_uses_symmetric_right_endpoint():
    edges = colon_edges(0.0, 0.1, 300.0)
    assert len(edges) == 3001 and edges[2453] == 300 - 547 * 0.1


def test_fractional_reference_labels_are_rejected_instead_of_truncated():
    seed = pd.DataFrame(
        {
            "UnitID": ["a", "b"],
            "Label": [1.5, 2.0],
            "Provenance": ["independent", "independent"],
        }
    )
    with pytest.raises(ValueError, match="exactly 1 or 2"):
        classify_with_seeds(np.array([[0.0], [1.0]]), ["a", "b"], seed)
