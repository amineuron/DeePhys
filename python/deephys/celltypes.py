"""Unlabelled phenotype mixtures and reference-seeded cell-type inference."""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .graphs import fuzzy_knn_graph, propagate_labels


def waveform_mixtures(table, *, seed=42, confidence=0.99):
    """Compare 1/2/3 Gaussian mixtures of width and trough-to-peak timing.

    Cluster identity is descriptive only. Broad/narrow waveforms are not used
    as experimentally validated E/I seeds or as labels for E/I ratios.
    """
    features = ["HalfWidth", "T2Pdelay"]
    raw = table[features].to_numpy(float)
    good = np.isfinite(raw).all(axis=1)
    output = pd.DataFrame(
        index=table.index,
        data={
            "WaveformCluster": 0,
            "MixtureMembership": np.nan,
            "ConfidentMixture": False,
        },
    )
    if good.sum() < 20:
        return output, pd.DataFrame(), {}
    scaler = StandardScaler()
    X = scaler.fit_transform(raw[good])
    fits = []
    summary = []
    for k in [1, 2, 3]:
        model = GaussianMixture(
            k, covariance_type="full", reg_covar=1e-6, n_init=10, random_state=seed
        ).fit(X)
        fits.append(model)
        summary.append(
            dict(
                Components=k,
                BIC=model.bic(X),
                AIC=model.aic(X),
                Converged=bool(model.converged_),
            )
        )
    model = fits[int(np.argmin([r["BIC"] for r in summary]))]
    # Shared cluster order by increasing physical half-width, never E/I tags.
    means = scaler.inverse_transform(model.means_)
    order = np.argsort(means[:, 0])
    ranks = np.empty(len(order), int)
    ranks[order] = np.arange(1, len(order) + 1)
    probability = model.predict_proba(X)
    assignments = ranks[probability.argmax(axis=1)]
    output.loc[good, "WaveformCluster"] = assignments
    output.loc[good, "MixtureMembership"] = probability.max(axis=1)
    output.loc[good, "ConfidentMixture"] = probability.max(axis=1) >= confidence
    details = dict(
        selected_components=int(model.n_components),
        confidence_threshold=confidence,
        fit_units=int(good.sum()),
        means_ms=means[order].tolist(),
        interpretation="Unlabelled waveform phenotypes; membership is not cell-type confidence",
    )
    return output, pd.DataFrame(summary), details


def classify_with_seeds(X, unit_ids, seeds, *, neighbors=50, min_score=0.99):
    """Clamped graph inference with provenance-annotated independent seeds.

    seeds columns: UnitID, Label (1=E/2=I), Provenance. Unknown components and
    low-score units stay 0. Graph scores are not calibrated probabilities.
    """
    required = {"UnitID", "Label", "Provenance"}
    if not required <= set(seeds):
        raise ValueError("seed table requires UnitID, Label, Provenance")
    if (
        seeds.UnitID.duplicated().any()
        or seeds.Provenance.isna().any()
        or seeds.Provenance.astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "unique seed identities and nonempty independent provenance required"
        )
    ids = pd.Index(unit_ids)
    positions = ids.get_indexer(seeds.UnitID)
    if np.any(positions < 0):
        raise ValueError("seed not present in feature matrix")
    if not seeds.Label.isin([1, 2]).all():
        raise ValueError("seed labels must be exactly 1 or 2")
    labels = seeds.Label.to_numpy(int)
    if set(labels) != {1, 2}:
        raise ValueError("both independently supported E and I seeds are required")
    W = fuzzy_knn_graph(X, neighbors)
    propagation = propagate_labels(W, positions, labels)
    score = propagation.scores.max(axis=1)
    prediction = propagation.scores.argmax(axis=1) + 1
    prediction[propagation.unreached | (score < min_score)] = 0
    prediction[positions] = labels
    return (
        pd.DataFrame(
            dict(
                UnitID=unit_ids,
                Label=prediction,
                GraphScoreE=propagation.scores[:, 0],
                GraphScoreI=propagation.scores[:, 1],
                Unreached=propagation.unreached,
                IsIndependentSeed=np.isin(np.arange(len(ids)), positions),
            )
        ),
        W,
        propagation,
    )
