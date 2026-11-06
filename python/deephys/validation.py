"""Validate ported features against completed local MATLAB outputs."""

from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .features import (
    activity_features,
    autocorrelogram,
    waveform_shape,
    harmonize,
    smooth_acg,
)
from .graphs import global_normalize, fuzzy_knn_graph, weight_feature_groups


def validate_native(native_root, output, *, limit=None):
    root = Path(native_root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    overall_start = time.monotonic()
    statuses = sorted((root / "native/primary_300s").glob("*/core_STATUS.json"))
    if limit is not None:
        statuses = statuses[:limit]
    for stfile in statuses:
        status = json.loads(stfile.read_text())
        key = stfile.parent.name
        if status["status"] not in ("complete", "partial"):
            continue
        checkpoint = out / (key + ".json")
        if checkpoint.exists():
            previous = json.loads(checkpoint.read_text())
            if previous.get("passed"):
                records.append(previous)
                continue
        start = time.monotonic()
        folder = stfile.parent
        d = loadmat(folder / "native_plot_data.mat", simplify_cells=True)
        table = pd.read_csv(folder / "UnitFeatures.csv")
        ids = np.atleast_1d(d["original_unit_id"]).astype(int)
        n = len(ids)
        wf = np.asarray(d["waveforms"]).reshape(-1, n)
        acg = np.asarray(d["acg"]).reshape(-1, n)
        trains = [
            np.atleast_1d(t).astype(float) for t in np.atleast_1d(d["spike_times"])
        ]
        np.testing.assert_array_equal(ids, table.OriginalUnitID)
        comparisons = {}
        failures = []

        def compare(name, a, b, *, rtol=1e-8, atol=1e-10):
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            valid = np.isfinite(a) & np.isfinite(b)
            differences = np.abs(a[valid] - b[valid])
            passed = bool(
                a.shape == b.shape
                and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            )
            comparisons[name] = {
                "passed": passed,
                "compared_values": int(a.size),
                "max_abs_error": float(differences.max(initial=0)),
                "rtol": rtol,
                "atol": atol,
            }
            if not passed:
                failures.append(name)

        # All selected units, not only representative/low-spike examples.
        activity = [activity_features(t, float(d["duration"])) for t in trains]
        shape = [waveform_shape(wf[:, i], float(d["fs"])) for i in range(n)]
        for family, values in [("activity", activity), ("waveform", shape)]:
            for feature in values[0]:
                if feature in table:
                    compare(
                        family + "/" + feature,
                        [v[feature] for v in values],
                        table[feature].to_numpy(),
                    )
        # Sample low/middle/high spike counts and spread IDs for exact ACG counts.
        order = np.argsort([len(t) for t in trains])
        selected = np.unique(
            np.r_[
                order[0],
                order[len(order) // 2],
                order[-1],
                np.linspace(0, n - 1, min(n, 5), dtype=int),
            ]
        )
        for i in selected:
            raw = autocorrelogram(trains[i])
            norm = raw / max(raw.max(), 1)
            compare("ACG/" + str(ids[i]), norm, acg[:, i], rtol=0, atol=0)
        graph_file = folder / "unlabeled_graph.mat"
        if graph_file.exists():
            graph = loadmat(graph_file, simplify_cells=True)
            smoothed = smooth_acg(acg, float(graph["smoothing_ms"]) / 0.5)
            raw, names, aligned, norm = harmonize(wf, smoothed, float(d["fs"]))
            compare(
                "harmonized_waveforms",
                aligned,
                graph["harmonized_waveforms"],
                atol=1e-8,
            )
            compare("harmonized_ACGs", norm, graph["harmonized_acgs"], atol=1e-8)
            # Native single-recording group z-score (constant columns -> NaN).
            missing = np.isnan(raw)
            with np.errstate(divide="ignore", invalid="ignore"):
                grouped = (raw - np.nanmean(raw, axis=0)) / np.nanstd(
                    raw, axis=0, ddof=1
                )
            grouped[missing] = 0
            X, normalization = global_normalize(grouped)
            compare("normalized_classifier_features", X, graph["X"], atol=1e-7)
            if X.shape == graph["X"].shape:
                native_graph = (
                    graph["G"].toarray()
                    if hasattr(graph["G"], "toarray")
                    else graph["G"]
                )
                trimmed_names = np.asarray(names)[normalization["keep"]]
                got = fuzzy_knn_graph(
                    weight_feature_groups(X, trimmed_names), min(50, n - 1)
                ).toarray()
                compare("fuzzy_graph_weights", got, native_graph, atol=1e-7)
        row = dict(
            key=key,
            units=n,
            passed=not failures,
            failures=failures,
            comparisons=comparisons,
            runtime_s=time.monotonic() - start,
        )
        checkpoint.write_text(json.dumps(row, indent=2), encoding="utf-8")
        records.append(row)
        summary = dict(
            wells_checked=len(records),
            units_checked=sum(r["units"] for r in records),
            all_passed=all(r["passed"] for r in records),
            runtime_s=time.monotonic() - overall_start,
            records=records,
        )
        (out / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: row[k] for k in ("key", "units", "passed", "failures", "runtime_s")}
            ),
            flush=True,
        )
    return records
