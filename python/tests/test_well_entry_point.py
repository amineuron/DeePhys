from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import savemat
from deephys.report import publish_wells


def test_per_well_entry_point_preserves_explicit_original_ids(tmp_path):
    g = json.loads(
        (Path(__file__).parent / "fixtures/matlab_synthetic_reference.json").read_text()
    )
    root = tmp_path / "input"
    folder = root / "native/primary_300s/synthetic"
    folder.mkdir(parents=True)
    times = np.empty(5, dtype=object)
    times[:] = [np.array(t) for t in g["times"]]
    ids = np.array([10, 20, 30, 40, 50])
    savemat(
        folder / "native_plot_data.mat",
        dict(
            original_unit_id=ids,
            waveforms=g["waveforms"],
            acg=g["acgs"],
            fs=10000.0,
            duration=3.0,
            spike_times=times,
            burst_start=np.array([]),
            burst_end=np.array([]),
        ),
    )
    pd.DataFrame(
        dict(
            OriginalUnitID=ids,
            HalfWidth=[0.1, 0.2, 0.3, 0.4, 0.5],
            T2Pdelay=[0.3, 0.4, 0.5, 0.6, 0.7],
        )
    ).to_csv(folder / "UnitFeatures.csv", index=False)
    out = tmp_path / "output"
    out.mkdir()
    pd.DataFrame(
        dict(
            RecordingID=["synthetic"] * 5,
            OriginalUnitID=ids,
            WaveformCluster=[1, 1, 2, 2, 0],
        )
    ).to_csv(out / "waveform_mixture_units.csv")
    published = publish_wells(root, out)
    assert published == ["synthetic"]
    assert (out / "wells/synthetic/index.html").is_file()
    assert json.loads((out / "WELLS_PROGRESS.json").read_text())["published"] == 1
