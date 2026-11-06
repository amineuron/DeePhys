"""Grouped phenotype prediction and label-free dimensionality reduction.

scikit-learn models/embeddings are independently implemented counterparts;
their stochastic predictions are not asserted to be MATLAB-identical.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def embeddings(X, *, seed=42, include_tsne=True, include_umap=True):
    """Label-free median imputation, constant-column removal and z-scoring."""
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
        ]
    )
    z = prep.fit_transform(X)
    if min(z.shape) < 2:
        raise ValueError("at least 2 variable features/observations required")
    results = {
        "PCA": PCA(
            n_components=2, svd_solver="randomized", random_state=seed
        ).fit_transform(z)
    }
    reduced = PCA(
        n_components=min(50, z.shape[1], len(z) - 1), random_state=seed
    ).fit_transform(z)
    if include_tsne and len(z) >= 5:
        results["tSNE"] = TSNE(
            n_components=2,
            perplexity=min(30, max(2, (len(z) - 1) / 3)),
            random_state=seed,
            init="pca",
            learning_rate="auto",
            n_jobs=1,
        ).fit_transform(reduced)
    if include_umap:
        try:
            import umap

            results["UMAP"] = umap.UMAP(
                n_components=2,
                n_neighbors=min(50, len(z) - 1),
                random_state=seed,
                n_jobs=1,
            ).fit_transform(z)
        except ImportError:
            pass
    return results, prep


def grouped_prediction(
    X, y, groups, *, task="classification", trees=500, seed=42, n_jobs=2
):
    """Hold an entire supplied group out; fit imputation/scaling on training only."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    X = pd.DataFrame(X)
    if len(np.unique(groups)) < 2:
        raise ValueError("at least two independent CV groups required")
    rows = []
    metrics = []
    importance = []
    models = []
    for fold, (train, test) in enumerate(LeaveOneGroupOut().split(X, y, groups), 1):
        if task == "classification":
            if len(np.unique(y[train])) < 2:
                raise ValueError("training fold lacks one class")
            estimator = RandomForestClassifier(
                n_estimators=trees,
                random_state=seed + fold,
                n_jobs=n_jobs,
                max_features="sqrt",
            )
        elif task == "regression":
            estimator = RandomForestRegressor(
                n_estimators=trees,
                random_state=seed + fold,
                n_jobs=n_jobs,
                max_features=1 / 3,
            )
        else:
            raise ValueError("unknown prediction task")
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", estimator),
            ]
        )
        model.fit(X.iloc[train], y[train])
        pred = model.predict(X.iloc[test])
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("CV group leakage")
        metric = dict(
            Fold=fold,
            TestGroups=",".join(map(str, np.unique(groups[test]))),
            TrainN=len(train),
            TestN=len(test),
        )
        if task == "classification":
            metric.update(
                Accuracy=accuracy_score(y[test], pred),
                BalancedAccuracy=balanced_accuracy_score(y[test], pred),
                MacroF1=f1_score(y[test], pred, average="macro", zero_division=0),
            )
        else:
            metric.update(
                MSE=mean_squared_error(y[test], pred), R2=r2_score(y[test], pred)
            )
        metrics.append(metric)
        rows.extend(
            dict(
                Row=int(i),
                Fold=fold,
                TestGroup=str(groups[i]),
                Observed=obs,
                Predicted=p,
            )
            for i, obs, p in zip(test, y[test], pred)
        )
        importance.extend(
            dict(Fold=fold, Feature=f, TrainingImpurityImportance=float(v))
            for f, v in zip(X.columns, estimator.feature_importances_)
        )
        models.append(model)
    return pd.DataFrame(rows), pd.DataFrame(metrics), pd.DataFrame(importance), models
