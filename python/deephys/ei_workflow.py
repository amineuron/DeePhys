"""End-to-end E/I reports when independent reference labels are available."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .celltypes import classify_with_seeds
from .features import classifier_features
from .graphs import weight_feature_groups
from .ei import (
    compute_activity,
    detect_ei_states,
    population_correlations,
    burst_cutouts,
)


def build_ei_reports(native_root, labels_csv, output):
    from .report import setup_style, page, save_figure, figure_html, TEAL, GREY
    import matplotlib.pyplot as plt

    setup_style()
    root = Path(native_root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_csv(labels_csv)
    required = {"RecordingID", "OriginalUnitID", "Label", "Provenance"}
    if not required <= set(seeds):
        raise ValueError(
            "labels require RecordingID, OriginalUnitID, Label, Provenance"
        )
    if seeds.duplicated(["RecordingID", "OriginalUnitID"]).any():
        raise ValueError("duplicate reference unit keys")
    records = []
    for key, refs in seeds.groupby("RecordingID", sort=True):
        source = root / "native/primary_300s" / key
        dest = out / key
        dest.mkdir(exist_ok=True)
        if not (source / "native_plot_data.mat").is_file():
            raise ValueError("missing source well: " + key)
        d = loadmat(source / "native_plot_data.mat", simplify_cells=True)
        ids = np.atleast_1d(d["original_unit_id"]).astype(int)
        n = len(ids)
        original = pd.Index(ids)
        positions = original.get_indexer(refs.OriginalUnitID)
        if np.any(positions < 0):
            raise ValueError("reference ID not in source QC selection")
        if set(refs.Label) != {1, 2}:
            records.append(
                dict(
                    RecordingID=key,
                    status="pending both independent seed classes",
                    E=0,
                    I=0,
                    Unknown=n,
                )
            )
            continue
        wf = np.asarray(d["waveforms"]).reshape(-1, n)
        acg = np.asarray(d["acg"]).reshape(-1, n)
        status = source / "unlabeled_STATUS.json"
        sigma = (
            float(json.loads(status.read_text())["acg_smoothing_ms"])
            if status.is_file()
            else 0.0
        )
        X, names, _, _ = classifier_features(
            wf, acg, float(d["fs"]), smoothing_ms=sigma
        )
        unit_ids = np.array([key + "::" + str(i) for i in ids])
        refs = refs.copy()
        refs["UnitID"] = unit_ids[positions]
        classified, _, diag = classify_with_seeds(
            weight_feature_groups(X, names), unit_ids, refs
        )
        classified.insert(0, "OriginalUnitID", ids)
        classified.to_csv(dest / "unit_labels_and_scores.csv", index=False)
        y = classified.Label.to_numpy()
        trains = [np.atleast_1d(t) for t in np.atleast_1d(d["spike_times"])]
        activity = compute_activity(trains, y, float(d["duration"]))
        state, floor = detect_ei_states(activity)
        correlations = population_correlations(activity)
        cutouts = burst_cutouts(activity)
        pd.DataFrame(
            dict(
                Time_s=activity.x,
                E_Hz=activity.E_hz,
                I_Hz=activity.I_hz,
                ClassifiedTotal_Hz=activity.total_hz,
                MeanRate_I_over_E=activity.ratio,
                E_dominant_state=state,
            )
        ).to_csv(dest / "activity.csv", index=False)
        pd.DataFrame(
            dict(
                OriginalUnitID=ids,
                Correlation_I=correlations[:, 0],
                Correlation_E=correlations[:, 1],
            )
        ).to_csv(dest / "leave_one_out_correlations.csv", index=False)
        np.savez_compressed(
            dest / "all_qualifying_burst_cutouts.npz",
            **{k: v for k, v in cutouts.items() if k != "method"},
        )
        fig, axes = plt.subplots(
            3, 1, figsize=(7, 5), sharex=True, layout="constrained"
        )
        axes[0].plot(activity.x, activity.E_hz, c=TEAL, lw=0.8, label="Inferred E")
        axes[0].plot(activity.x, activity.I_hz, c=GREY, lw=0.8, label="Inferred I")
        axes[0].set(
            ylabel="Mean rate (Hz/unit)", title="Reference-seeded cell-type activity"
        )
        axes[0].legend(frameon=False)
        axes[1].plot(activity.x, activity.ratio, c=TEAL, lw=0.8)
        axes[1].set(ylabel="Mean I rate / E rate")
        axes[2].fill_between(activity.x, 0, state, color=GREY, step="post")
        axes[2].set(xlabel="Time (s)", ylabel="E-dominant state", yticks=[0, 1])
        save_figure(fig, dest, "ei_activity")
        body = (
            '<nav><a href="../index.html">E/I wells</a></nav><h1>'
            + key
            + " | Reference-seeded E/I analysis</h1>"
        )
        body += figure_html(
            "ei_activity",
            "E/I activity and candidate states",
            "Labels combine independent supplied seeds with graph inference above the explicit score threshold. Graph scores are not calibrated biological probabilities. Unknown units are excluded; rates are per class member. Candidate E-dominant states are distinct from ISINJoint network bursts.",
        )
        body += (
            "<p>Unknown units: "
            + str(int((y == 0).sum()))
            + " of "
            + str(n)
            + ". Independent seeds: "
            + str(len(refs))
            + ".</p>"
        )
        (dest / "index.html").write_text(page(key + " E/I", body), encoding="utf-8")
        records.append(
            dict(
                RecordingID=key,
                status="complete",
                E=int((y == 1).sum()),
                I=int((y == 2).sum()),
                Unknown=int((y == 0).sum()),
                IndependentSeeds=len(refs),
                PropagationConverged=diag.converged,
            )
        )
    pd.DataFrame(records).to_csv(out / "coverage.csv", index=False)
    body = "<h1>Reference-seeded E/I analysis</h1><p>Only wells with both independently supported seed classes can be processed. A missing class is not interpreted as absent neurons.</p><ul>"
    for r in records:
        key = r["RecordingID"]
        body += (
            "<li>"
            + (
                '<a href="' + key + '/index.html">' + key + "</a>"
                if r["status"] == "complete"
                else key + " — " + r["status"]
            )
            + "</li>"
        )
    (out / "index.html").write_text(
        page("E/I analysis", body + "</ul>"), encoding="utf-8"
    )
    return records
