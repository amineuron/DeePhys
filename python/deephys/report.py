"""Offline figures and tables for the four post-sorting tutorial sections."""

from pathlib import Path
from datetime import datetime, timezone
import html, json, gc
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import adjusted_rand_score
from .store import FeatureStore
from .phenotype import embeddings, grouped_prediction
from .celltypes import waveform_mixtures
from .graphs import fuzzy_knn_graph, weight_feature_groups, leiden_communities
from .numeric import bin_spikes
from .features import classifier_features

TEAL = "#1f6f5f"
GREY = "#888883"
INK = "#171717"
CSS = """body{font:16px/1.5 Arial,sans-serif;max-width:1180px;padding:24px;margin:auto;color:#171717;background:white}a{color:#1f6f5f}nav{display:flex;flex-wrap:wrap;gap:20px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}th{border-top:1px solid #888}.num{text-align:right}.note{font-size:14px;color:#555}select{font:inherit;padding:8px}h2{margin-top:32px}iframe{width:100%;height:1100px;border:1px solid #ddd}code{overflow-wrap:anywhere}"""


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def page(title, body):
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + html.escape(title)
        + "</title><style>"
        + CSS
        + "</style><body>"
        + body
        + "</body></html>"
    )


def save_figure(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=210)
    fig.savefig(folder / (name + ".pdf"))
    plt.close(fig)


def figure_html(name, title, caption):
    return f'<h2>{html.escape(title)}</h2><a href="{name}.pdf">PDF</a><img loading="lazy" src="{name}.png" alt="{html.escape(title)}"><p class="note">{html.escape(caption)}</p>'


def report_index(out, store, stages, wells):
    stamp = datetime.now(timezone.utc).isoformat()
    b = '<h1>DeePhys Python | Post-sorting analysis</h1><nav><a href="feature_exploration.html">Feature exploration</a><a href="phenotype.html">Phenotype models</a><a href="celltypes.html">Cell-type phenotypes</a><a href="ei.html">E/I analysis and label requirements</a><a href="validation/SUMMARY.json">MATLAB agreement</a></nav>'
    b += f"<p>{len(store.recordings)} well-days; {len(store.units):,} selected-unit observations. Features are from the native DeePhys exports; independently ported primitives have separate MATLAB agreement results. All primary windows are 300 s.</p>"
    b += "<p>Blinded A = wells 1–3; B = wells 4–6. Genotype mapping is unconfirmed. N1 and N3 are two biological replicates; units and repeated days are not independent biological samples.</p>"
    b += (
        "<h2>Live stage status</h2><table><tr><th>Section</th><th>Status</th></tr>"
        + "".join(
            "<tr><td>" + html.escape(k) + "</td><td>" + html.escape(v) + "</td></tr>"
            for k, v in stages.items()
        )
        + "</table>"
    )
    b += (
        '<h2>Per-well figures</h2><label>Well-day <select id="well">'
        + "".join(
            '<option value="' + html.escape(k) + '">' + html.escape(k) + "</option>"
            for k in wells
        )
        + '</select></label> <a id="open">Open selected well</a><iframe id="frame" title="Selected Python well report"></iframe>'
    )
    b += """<script>const w=document.getElementById('well'),f=document.getElementById('frame'),a=document.getElementById('open');function show(){let p='wells/'+w.value+'/index.html';f.src=p;a.href=p;location.hash=w.value}let key=decodeURIComponent(location.hash.slice(1));if([...w.options].some(x=>x.value===key))w.value=key;w.onchange=show;if(w.options.length)show();else f.hidden=true;</script>"""
    b += (
        '<h2>Scope and provenance</h2><p><a href="run_manifest.json">Run manifest</a> · <a href="feature_columns.json">Model input columns</a> · <a href="well_coverage.csv">Well coverage</a>. Native ISINJoint event boundaries are imported, not relabelled as a Python detector. Mixture clusters and Leiden communities have no validated E/I identities.</p><p class="note">Updated '
        + stamp
        + ". Reload to see newly published sections. Source inputs and NAS locations are read-only.</p>"
    )
    (out / "index.html").write_text(page("DeePhys Python", b), encoding="utf-8")
    (out / "PROGRESS.json").write_text(
        json.dumps(
            dict(
                utc=stamp,
                stages=stages,
                published_wells=len(wells),
                source_wells=len(store.recordings),
                unit_observations=len(store.units),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def build_feature_exploration(out, store, X, md):
    body = '<nav><a href="index.html">Python all wells</a></nav><h1>Feature exploration</h1>'
    completeness = X.notna().mean().sort_values()
    completeness.rename("FractionAvailable").to_csv(out / "feature_availability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), layout="constrained")
    selected = completeness.head(12).iloc[::-1]
    axes[0].barh(selected.index, selected.values, color=TEAL)
    axes[0].set(
        xlim=(0, 1),
        xlabel="Fraction of observations",
        title="Least complete input features",
    )
    counts = [
        sum(
            "Waveform" in c or "AUC" in c or c in ("HalfWidth", "T2Pdelay", "T2Pratio")
            for c in X
        ),
        sum("ACG" in c for c in X),
    ]
    counts.append(len(X.columns) - sum(counts))
    axes[1].barh(["Waveform", "ACG", "Other scalar features"], counts, color=TEAL)
    axes[1].set(xlabel="Feature columns", title="Available numerical feature families")
    save_figure(fig, out, "feature_availability")
    body += figure_html(
        "feature_availability",
        "Feature availability",
        "Missing values remain missing until a training-fold imputer is fit. Stage-dependent STTC columns and identifiers are excluded from phenotype predictors.",
    )
    core = [
        "FiringRate",
        "MeanInterSpikeInterval",
        "CVInterSpikeInterval",
        "CV2InterSpikeInterval",
        "LocalVariation",
        "RevisedLocalVariation",
        "FanoFactor",
        "PartialAutocorrelation",
        "ISIBimodality",
        "HalfWidth",
        "T2Pdelay",
        "T2Pratio",
        "Asymmetry",
        "AUC_trough",
        "RegularityFrequency",
        "RegularityMagnitude",
    ]
    variances = pd.Index([c for c in core if c in X])
    corr = X[variances].corr(method="spearman")
    corr.to_csv(out / "scalar_feature_spearman.csv")
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, rasterized=True)
    labels = [x.replace("InterSpikeInterval", "ISI") for x in variances]
    ax.set(
        xticks=range(len(labels)),
        xticklabels=labels,
        yticks=range(len(labels)),
        yticklabels=labels,
        title="Scalar feature redundancy",
    )
    ax.tick_params(axis="x", rotation=90, labelsize=8.5)
    fig.colorbar(im, ax=ax, label="Spearman correlation", shrink=0.7)
    save_figure(fig, out, "feature_correlations")
    body += figure_html(
        "feature_correlations",
        "Scalar feature relationships",
        "Predeclared activity and waveform scalars; exploratory unit-level correlation without treating cells as independent biological replicates. The full predictor column list is retained. ISI means inter-spike interval.",
    )
    body += '<p><a href="feature_availability.csv">All feature availability</a> · <a href="unit_metadata.csv">Unit identities/metadata</a> · <a href="recording_features.csv">Recording feature matrix</a> · <a href="unit_features.csv.gz">Full unit feature matrix (gzip CSV)</a></p>'
    (out / "feature_exploration.html").write_text(
        page("Feature exploration", body), encoding="utf-8"
    )


def plot_embeddings(out, level, results, metadata):
    fig, axes = plt.subplots(
        1, len(results), figsize=(7, 2.8), layout="constrained", squeeze=False
    )
    for ax, (method, xy) in zip(axes[0], results.items()):
        for rep, marker in [("N1", "o"), ("N3", "^")]:
            for group, color in [("A", TEAL), ("B", GREY)]:
                take = metadata.Replicate.eq(rep) & metadata.WellGroup.eq(group)
                ax.scatter(
                    xy[take, 0],
                    xy[take, 1],
                    s=4 if level == "unit" else 20,
                    c=color,
                    marker=marker,
                    alpha=0.4 if level == "unit" else 0.8,
                    label=rep + "/" + group,
                    rasterized=True,
                )
        ax.set(xlabel=method + " 1", ylabel=method + " 2", title=method)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=4, loc="outside lower center", frameon=False, fontsize=8.5
    )
    save_figure(fig, out, level + "_embeddings")


def per_well(out, native_root, key, frame, mixtures, *, communities=True):
    src = Path(native_root) / "native/primary_300s" / key
    dest = out / "wells" / key
    dest.mkdir(parents=True, exist_ok=True)
    completed = dest / "STATUS.json"
    if completed.exists() and json.loads(completed.read_text()).get("complete"):
        return
    d = loadmat(src / "native_plot_data.mat", simplify_cells=True)
    n = len(frame)
    ids = np.atleast_1d(d["original_unit_id"]).astype(int)
    n = len(ids)
    times = [np.atleast_1d(x) for x in np.atleast_1d(d["spike_times"])]
    duration = float(d["duration"])
    wf = np.asarray(d["waveforms"]).reshape(-1, n)
    acg = np.asarray(d["acg"]).reshape(-1, n)
    body = (
        '<nav><a href="../../index.html" target="_top">Python all wells</a></nav><h1>'
        + key
        + "</h1><p>"
        + str(n)
        + " native-QC selected units; window "
        + str(duration)
        + " s. Cell-type identity remains unassigned.</p>"
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 4.6),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].eventplot(times, colors=INK, linewidths=0.25, linelengths=0.7)
    axes[0].set(ylabel="Selected unit row", title="All selected sorted units")
    counts, edges = bin_spikes(times, duration, 0.1)
    axes[1].plot(edges[:-1], counts.sum(axis=0) / max(n, 1) / 0.1, c=TEAL, lw=0.8)
    for start, end in zip(
        np.atleast_1d(d["burst_start"]), np.atleast_1d(d["burst_end"])
    ):
        axes[1].axvspan(start, end, color=GREY, alpha=0.15, lw=0)
    axes[1].set(xlabel="Time (s)", ylabel="Mean rate (Hz/unit)", xlim=(0, duration))
    save_figure(fig, dest, "raster")
    body += figure_html(
        "raster",
        "Raster and population activity",
        "Raster, rates and binning use Python. Shading imports the selected native MATLAB ISINJoint event scale; it is not an independent Python burst detection result.",
    )
    selected = mixtures.loc[frame.index]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    for cluster, color, marker in [
        (1, TEAL, "o"),
        (2, GREY, "s"),
        (3, INK, "^"),
        (0, "#b6b6b0", "x"),
    ]:
        mask = selected.WaveformCluster.eq(cluster)
        if mask.any():
            axes[0].scatter(
                frame.loc[mask, "HalfWidth"],
                frame.loc[mask, "T2Pdelay"],
                s=10,
                c=color,
                marker=marker,
                alpha=0.5,
                label="Phenotype " + str(cluster),
                rasterized=True,
            )
    axes[0].set(
        xlabel="Half-width (ms)",
        ylabel="Trough-to-peak (ms)",
        title="Shared waveform mixture",
    )
    axes[0].legend(frameon=False, fontsize=8.5)
    axis_time = np.arange(len(wf)) / float(d["fs"]) * 1000
    axes[1].plot(axis_time, wf, c=GREY, alpha=0.07, lw=0.4, rasterized=True)
    axes[1].plot(axis_time, np.median(wf, axis=1), c=TEAL, lw=1.5)
    axes[1].set(
        xlabel="Template time (ms)",
        ylabel="Normalized amplitude",
        title="Measured reference waveforms",
    )
    save_figure(fig, dest, "waveform_phenotypes")
    body += figure_html(
        "waveform_phenotypes",
        "Waveform phenotypes",
        "A single pooled 1/2/3-component model is selected by BIC. Cluster numbers are shared across wells, ordered by width. These are not excitatory/inhibitory labels. All source-QC units are shown; mixture fitting excludes duplicate-review units.",
    )
    community_file = src / "unlabeled_graph.mat"
    info = {}
    if communities and community_file.exists():
        g = loadmat(community_file, simplify_cells=True)
        X, names, _, _ = classifier_features(
            wf, acg, float(d["fs"]), smoothing_ms=float(g["smoothing_ms"])
        )
        W = fuzzy_knn_graph(weight_feature_groups(X, names), min(50, len(X) - 1))
        labels, partitions, scores = leiden_communities(W, restarts=30)
        info = dict(
            communities=int(len(np.unique(labels))),
            ARI_vs_native=float(adjusted_rand_score(g["community"], labels)),
            graph_origin="Python harmonization, normalization and exact fuzzy graph",
        )
        pd.DataFrame(
            dict(
                OriginalUnitID=ids,
                PythonCommunity=labels,
                NativeCommunity=g["community"],
            )
        ).to_csv(dest / "community_method_comparison.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
        xy = g["embedding"]
        im = axes[0].scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels,
            cmap="Greys",
            s=10,
            edgecolors="black",
            linewidths=0.15,
            rasterized=True,
        )
        axes[0].set(
            xlabel="Native UMAP 1",
            ylabel="Native UMAP 2",
            title="Python Leiden on the feature graph",
        )
        fig.colorbar(im, ax=axes[0], label="Local phenotype community", shrink=0.7)
        agreements = np.array([adjusted_rand_score(labels, p) for p in partitions])
        ordered = np.sort(agreements)
        axes[1].step(
            ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", c=TEAL
        )
        axes[1].set(
            xlabel="Adjusted Rand index",
            ylabel="Fraction of Leiden restarts",
            title="30 Python restarts",
            ylim=(0, 1),
        )
        save_figure(fig, dest, "communities")
        body += figure_html(
            "communities",
            "Cell phenotype communities",
            "Python Leiden is displayed on the existing native embedding so partition changes can be compared without UMAP layout changes. Community numbers are local to this well. ARI versus native = "
            + f"{info['ARI_vs_native']:.3f}"
            + ". Agreement is algorithmic, not biological E/I validation.",
        )
    body += "<h2>E/I analysis</h2><p>Pending independently supported E/I seeds. No zero inhibitory fraction is reported from missing labels. Network burst patterns remain available above.</p>"
    (dest / "index.html").write_text(page(key, body), encoding="utf-8")
    completed.write_text(
        json.dumps(dict(complete=True, key=key, units=n, **info), indent=2),
        encoding="utf-8",
    )


def build_reports(native_root, output, *, trees=500, do_embeddings=True, do_wells=True):
    setup_style()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore.from_native(native_root)
    X, md = store.matrix()
    R, rmd = store.matrix("recording")
    stages = {
        "Feature exploration": "running",
        "Phenotype analysis": "queued",
        "Cell-type phenotypes": "queued",
        "E/I analysis": "awaiting independent seeds; synthetic numerical checks available",
    }
    published = []
    report_index(out, store, stages, published)
    md.to_csv(out / "unit_metadata.csv", index=False)
    rmd.to_csv(out / "recording_metadata.csv", index=False)
    X.to_csv(out / "unit_features.csv.gz", index=False, compression="gzip")
    R.to_csv(out / "recording_features.csv", index=False)
    (out / "feature_columns.json").write_text(
        json.dumps(
            dict(
                unit=list(X),
                recording=list(R),
                metadata_excluded=sorted(set(store.units) - set(X)),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    store.recordings[["RecordingID", "Replicate", "DIV", "Well", "WellGroup"]].to_csv(
        out / "well_coverage.csv", index=False
    )
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                **store.provenance,
                "python_implementation": "amineuron/DeePhys python-port",
                "source_files_read_only": True,
                "NAS_writes": 0,
                "native_full_feature_bridge": True,
                "raw_voltage_copied": False,
                "CV": "leave entire biological replicate out",
                "independent_biological_replicates": 2,
                "trees": trees,
                "E_I_labels_assigned": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_feature_exploration(out, store, X, md)
    stages["Feature exploration"] = "complete"
    mixtures, bic, details = waveform_mixtures(
        store.units.loc[~store.units.DuplicateReview]
    )
    mixtures = mixtures.reindex(store.units.index)
    mixtures["WaveformCluster"] = mixtures.WaveformCluster.fillna(0).astype(int)
    mixtures.to_csv(out / "waveform_mixture_units.csv")
    bic.to_csv(out / "waveform_mixture_BIC.csv", index=False)
    (out / "waveform_mixture_model.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    ax.plot(bic.Components, bic.BIC, "o-", c=TEAL)
    ax.set(
        xlabel="Number of Gaussian components",
        ylabel="BIC (lower is preferred)",
        xticks=[1, 2, 3],
        title="Waveform mixture complexity is selected from the data",
    )
    save_figure(fig, out, "mixture_model_selection")
    b = '<nav><a href="index.html">All wells</a></nav><h1>Cell-type phenotype analysis</h1>'
    b += figure_html(
        "mixture_model_selection",
        "Waveform mixture model selection",
        "Mixtures use native half-width and trough-to-peak features, excluding conservative duplicate-review units. Cluster membership is not an E/I identity. The seed-based graph propagation implementation has MATLAB reference tests; valid biological seeds have not been supplied.",
    )
    b += '<p><a href="waveform_mixture_units.csv">Per-unit mixture assignments</a> · <a href="waveform_mixture_BIC.csv">All BIC/AIC values</a></p>'
    (out / "celltypes.html").write_text(
        page("Cell-type phenotypes", b), encoding="utf-8"
    )
    (out / "ei.html").write_text(
        page(
            "E/I analysis",
            '<nav><a href="index.html">All wells</a></nav><h1>E/I analysis</h1><p>The Python activity, I/E ratios, E-dominant states, leave-one-out population correlations and STTC primitives are tested against synthetic MATLAB reference outputs. Real-data E/I quantification requires supported labels. The current waveform mixtures and Leiden communities remain unlabelled.</p><p>No E/I ratio, fraction or cell-type certainty is fabricated from sorting labels. All-unit network activity and the native ISINJoint burst boundaries are shown for every published well.</p>',
        ),
        encoding="utf-8",
    )
    stages["Cell-type phenotypes"] = (
        "pooled waveform mixture complete; per-well communities publishing"
    )
    report_index(out, store, stages, published)
    phenotype = '<nav><a href="index.html">All wells</a></nav><h1>Phenotype analysis</h1><p>Held-out folds are entire biological replicates. A/B predicts blinded numbered-well blocks; genotype identity remains unconfirmed. No random unit split is used.</p>'
    (out / "phenotype.html").write_text(
        page("Phenotype analysis", phenotype + "<p>Models running.</p>"),
        encoding="utf-8",
    )
    for level, data, metadata in [("recording", R, rmd), ("unit", X, md)]:
        if do_embeddings:
            stages["Phenotype analysis"] = level + " PCA/UMAP/tSNE running"
            report_index(out, store, stages, published)
            results, _ = embeddings(data)
            for method, xy in results.items():
                t = metadata.copy()
                t["Dim1"] = xy[:, 0]
                t["Dim2"] = xy[:, 1]
                t.to_csv(out / (level + "_" + method + ".csv"), index=False)
            plot_embeddings(out, level, results, metadata)
            phenotype += figure_html(
                level + "_embeddings",
                level.capitalize() + " feature projections",
                "Python/scikit-learn and umap-learn counterparts; stochastic coordinates are not asserted to be MATLAB-identical. All units are observations nested within two biological replicates.",
            )
        for target, task in [("WellGroup", "classification"), ("DIV", "regression")]:
            stages["Phenotype analysis"] = (
                level + " " + target + " " + task + " running"
            )
            report_index(out, store, stages, published)
            name = level + "_" + target
            predictions, metrics, importance, models = grouped_prediction(
                data, metadata[target], metadata.Replicate, task=task, trees=trees
            )
            predictions.to_csv(out / (name + "_predictions.csv"), index=False)
            metrics.to_csv(out / (name + "_metrics.csv"), index=False)
            importance.to_csv(out / (name + "_importance.csv"), index=False)
            del models
            gc.collect()
            fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
            if task == "classification":
                axes[0].plot(metrics.Fold, metrics.BalancedAccuracy, "o", c=TEAL)
                axes[0].axhline(0.5, c=GREY, ls="--", lw=0.7)
                axes[0].set(
                    xlabel="Held-out replicate fold",
                    ylabel="Balanced accuracy",
                    ylim=(0, 1),
                    title="Two biological-replicate folds",
                )
                cm = pd.crosstab(predictions.Observed, predictions.Predicted).reindex(
                    index=["A", "B"], columns=["A", "B"], fill_value=0
                )
                axes[1].imshow(cm, cmap="Greys", vmin=0)
                axes[1].set(
                    xticks=[0, 1],
                    xticklabels=["A", "B"],
                    yticks=[0, 1],
                    yticklabels=["A", "B"],
                    xlabel="Predicted block",
                    ylabel="Observed block",
                    title="Held-out predictions",
                )
                for i in range(2):
                    for j in range(2):
                        axes[1].text(
                            j,
                            i,
                            str(cm.iloc[i, j]),
                            ha="center",
                            va="center",
                            color="white"
                            if cm.iloc[i, j] > cm.to_numpy().max() / 2
                            else INK,
                        )
            else:
                for fold, p in predictions.groupby("Fold"):
                    axes[0].scatter(
                        p.Observed,
                        p.Predicted,
                        s=5,
                        c=TEAL if fold == 1 else GREY,
                        marker="o" if fold == 1 else "^",
                        alpha=0.35,
                        rasterized=True,
                    )
                lo = min(predictions.Observed.min(), predictions.Predicted.min())
                hi = max(predictions.Observed.max(), predictions.Predicted.max())
                axes[0].plot([lo, hi], [lo, hi], c=GREY, ls="--", lw=0.7)
                axes[0].set(
                    xlabel="Observed DIV (days)",
                    ylabel="Predicted DIV (days)",
                    title="Held-out age predictions",
                )
                axes[1].plot(metrics.Fold, metrics.R2, "o", c=TEAL)
                axes[1].axhline(0, c=GREY, ls="--", lw=0.7)
                axes[1].set(
                    xlabel="Held-out replicate fold",
                    ylabel="R²",
                    title="Negative values underperform the mean",
                )
            save_figure(fig, out, name)
            phenotype += figure_html(
                name,
                level.capitalize() + " " + target + " " + task,
                "500-tree random forests"
                if trees == 500
                else str(trees) + "-tree random forests",
            )
            phenotype += (
                '<p><a href="'
                + name
                + '_metrics.csv">Exact fold metrics</a> · <a href="'
                + name
                + '_predictions.csv">All held-out predictions</a> · <a href="'
                + name
                + '_importance.csv">Training impurity importance</a></p>'
            )
            (out / "phenotype.html").write_text(
                page("Phenotype analysis", phenotype), encoding="utf-8"
            )
    stages["Phenotype analysis"] = "complete; group-held-out predictions published"
    if do_wells:
        for key, frame in store.units.groupby("RecordingID", sort=True):
            per_well(out, native_root, key, frame, mixtures)
            published.append(key)
            stages["Cell-type phenotypes"] = (
                f"{len(published)} per-well reports complete"
            )
            report_index(out, store, stages, published)
    stages["Cell-type phenotypes"] = (
        "complete for available wells; E/I identity awaits independent seeds"
    )
    report_index(out, store, stages, published)
    return store
