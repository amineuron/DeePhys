# DeePhys Python companion

This branch adds Python post-sorting workflows alongside the original MATLAB
source. It is an alpha companion with function-level validation. No research
recordings or experimental labels are included.

## Installation and use

```console
git clone --branch python-port https://github.com/amineuron/DeePhys.git
cd DeePhys
pip install -e '.[embedding,communities,test]'
python -m pytest python/tests -q
deephys-python analyse --native-root /path/to/native_export --output /local/results
deephys-python validate --native-root /path/to/native_export --output /local/results/validation
```

Open `/local/results/index.html`. It links feature matrices, missingness and
correlations; PCA/UMAP/t-SNE; random-forest classification and age regression;
waveform mixtures and per-well Leiden communities; and E/I label requirements.
PNG, PDF and exact CSV outputs are retained. Analysis uses Python at runtime.

The native-export bridge reads `native/primary_300s/<recording>/` containing
`core_STATUS.json`, `UnitFeatures.csv`, `NetworkFeatures.csv`, and
`native_plot_data.mat`. These are data inputs, not a MATLAB runtime dependency.
The complete feature table and QC selection currently originate in MATLAB.
The ported spike/waveform primitives can also be used independently; unported
native features are not described as Python extractions.

## Feature exploration

```python
from deephys.store import FeatureStore

store = FeatureStore.from_native("/path/to/native_export")
X, metadata = store.matrix("unit")
recordings, recording_metadata = store.matrix("recording")
```

Metadata/IDs are separated from predictors. Missing values remain missing until
a training-fold imputer is fitted. Stage-dependent STTC columns are excluded
from incremental phenotype models. Duplicate-review units remain in source
tables and are excluded conservatively from model inputs. Numbered-well
trajectories require exact DIV matches; physical identity must be confirmed.

## Phenotype analysis

```python
from deephys.phenotype import embeddings, grouped_prediction

projection, preprocessing = embeddings(X)
predictions, metrics, importance, models = grouped_prediction(
    X, metadata.WellGroup, metadata.Replicate, task="classification", trees=500
)
age_predictions, age_metrics, age_importance, age_models = grouped_prediction(
    X, metadata.DIV, metadata.Replicate, task="regression", trees=500
)
```

The cohort runner holds out the entire biological replicate. Cells and repeated
days are not biological replicates. A/B numbered-well blocks are not assigned
genotypes. Training impurity importances differ from native held-out permutation
importances and are named accordingly. Python forests/embeddings/Leiden are
independent implementations; the same seed does not imply MATLAB-identical
stochastic results. N1's extra DIV22 also limits cross-replicate age prediction.

## Cell-type classification

```python
from deephys.features import classifier_features
from deephys.graphs import weight_feature_groups
from deephys.celltypes import classify_with_seeds

X, names, aligned, normalized_acgs = classifier_features(
    waveforms, acgs, sampling_rate, smoothing_ms=3.5
)
weighted = weight_feature_groups(X, names)
# seeds: UnitID, Label (1=E, 2=I), Provenance.
labels, graph, diagnostics = classify_with_seeds(weighted, unit_ids, seeds)
```

Clamped graph propagation follows the native method, including seed-count
normalization and unknown disconnected components. Scores are graph absorption
scores, not calibrated biological probabilities. The 0.99 score threshold is
an explicit companion API choice, not the native full confidence procedure.
Both independently supported E and I reference labels are required.

Without such references, the report fits unlabelled 1/2/3-component waveform
mixtures selected by BIC and phenotype communities. Broad/narrow waveforms
are not promoted to E/I ground truth. The mixture is a separate exploratory
method. The drug-response bootstrap, seed enrichment, composition estimates
and full native confidence calibration are not ported here.

## E/I activity and burst analysis

```python
from deephys.ei import (
    compute_activity,
    detect_ei_states,
    population_correlations,
    burst_cutouts,
)

# One supported label per unit: 0 unknown, 1 E, 2 I.
activity = compute_activity(spike_trains, labels, duration=300.0)
state, noise_floor = detect_ei_states(activity)
correlations = population_correlations(activity)
cutouts = burst_cutouts(activity)
```

With reference labels in a CSV containing `RecordingID`, `OriginalUnitID`,
`Label` (1=E, 2=I), and `Provenance`, the end-to-end entry point is:

```console
deephys-python ei --native-root /path/to/native_export --labels /local/seeds.csv --output /local/ei_results
```

Both independent seed classes are required within each processed well. The
command maps original IDs, runs clamped inference, writes all scores/unknown
labels, and produces activity/state figures, cutouts and correlations.

Activity is stored in spikes/bin; `E_hz`, `I_hz`, `total_hz` convert to Hz.
Unknown units do not enter E/I populations. I/E is undefined where E is zero.
Self-class correlations leave the unit out; a singleton class returns NaN
rather than the native fallback's trivial self-correlation of one.

`detect_ei_states` ports EIAnalyzer's E-dominant state detector. It is distinct
from all-unit ISINJoint network bursts. The report imports native ISINJoint
boundaries. Full iterative burst-template acceptance/alignment and native
burst/null feature suites are not yet ported. `rate_matched_null` is a separately
named supplemental rate-stratified label-permutation null.

## Validation and numerical conventions

The synthetic JSON fixture was executed in MATLAB R2023b at upstream commit
`5efdcb82fae3700a179bde10105d9d5dc7743d52`. It contains no research data.
Tests cover activity, seven waveform shape features, ACGs, makima harmonization,
center-preserving smoothing, normalization, fuzzy graphs, clamped propagation,
E/I activity/ratios/states, leave-one-out correlations and empirical STTC.
Most tolerances are 1e-10 relative / 1e-12 absolute; normalized almost-flat
makima derivatives allow 1e-10 absolute.

Real-data validation stays local: every selected unit's supported scalar
features, sampled ACGs, all harmonized features and fuzzy weights are compared.
This does not validate biological identity or unported procedures.

MATLAB's colon constructs histogram edges symmetrically from their endpoints.
Its CCG wrapper quantizes times, uses signed half-bin ties, and flips counts.
These details affect exact sample-grid boundary cases and are preserved.

## Read-only inputs and storage

The CLI rejects input-root mutations and network writes. Outputs must be local.
Phy loading uses read-only NumPy maps, does not execute `params.py`, preserves
original IDs, and requires explicit duration and amplitude calibration. No
continuous raw voltage is copied or source file deleted. The API returns fitted
models; the report persists predictions, parameters and importances rather
than duplicating large serialized forests.

Original author/method citations remain in the repository README. The machine
readable [coverage file](COVERAGE.json) records current limitations.
