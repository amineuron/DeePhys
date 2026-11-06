from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import savemat
from deephys.ei_workflow import build_ei_reports


def test_complete_ei_entry_point_uses_original_seed_ids_and_keeps_unknowns(tmp_path):
    g = json.loads(
        (Path(__file__).parent / "fixtures/matlab_synthetic_reference.json").read_text()
    )
    root = tmp_path / "input"
    folder = root / "native/primary_300s/synthetic"
    folder.mkdir(parents=True)
    trains = np.empty(5, dtype=object)
    trains[:] = [np.array(t) for t in g["times"]]
    ids = np.array([10, 20, 30, 40, 50])
    savemat(
        folder / "native_plot_data.mat",
        dict(
            original_unit_id=ids,
            waveforms=g["waveforms"],
            acg=g["acgs"],
            fs=10000.0,
            duration=3.0,
            spike_times=trains,
        ),
    )
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        dict(
            RecordingID=["synthetic"] * 4,
            OriginalUnitID=ids[:4],
            Label=[1, 1, 2, 2],
            Provenance=["synthetic independent reference"] * 4,
        )
    ).to_csv(labels, index=False)
    out = tmp_path / "output"
    records = build_ei_reports(root, labels, out)
    assert records[0]["status"] == "complete"
    assert records[0]["IndependentSeeds"] == 4
    exported = pd.read_csv(out / "synthetic/unit_labels_and_scores.csv")
    np.testing.assert_array_equal(exported.OriginalUnitID, ids)
    np.testing.assert_array_equal(exported.Label.iloc[:4], [1, 1, 2, 2])
    assert (out / "synthetic/ei_activity.pdf").is_file()
    assert (out / "index.html").is_file()
