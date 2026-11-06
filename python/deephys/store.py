"""Feature matrices with metadata kept outside numerical model predictors."""

from dataclasses import dataclass
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

METADATA = {
    "OriginalUnitID",
    "TemplateID",
    "UnitID",
    "RecordingID",
    "Replicate",
    "DIV",
    "Well",
    "WellGroup",
    "key",
    "CultureID",
    "Genotype",
    "Mutation",
    "ReferenceElectrode",
    "ReferenceChannel",
    "ChannelID",
    "X_um",
    "Y_um",
    "SamplingRate",
    "Duration",
    "RecordingDuration",
}


@dataclass
class FeatureStore:
    units: pd.DataFrame
    recordings: pd.DataFrame
    provenance: dict

    @classmethod
    def from_native(cls, root):
        """Read completed native per-recording CSVs; never modify inputs."""
        root = Path(root)
        units = []
        recordings = []
        statuses = {}
        for status_file in sorted(
            (root / "native" / "primary_300s").glob("*/core_STATUS.json")
        ):
            status = json.loads(status_file.read_text(encoding="utf-8-sig"))
            if status["status"] not in ("complete", "partial", "empty_qc"):
                continue
            folder = status_file.parent
            key = folder.name
            match = re.fullmatch(r"(N\d+)_DIV(\d+)__well(\d+)", key)
            if not match:
                continue
            rep, div, well = match[1], int(match[2]), int(match[3])
            md = dict(
                RecordingID=key,
                Replicate=rep,
                DIV=div,
                Well=well,
                WellGroup="A" if well <= 3 else "B",
            )
            statuses[key] = status
            if (folder / "UnitFeatures.csv").exists():
                frame = pd.read_csv(
                    folder / "UnitFeatures.csv", low_memory=False
                ).copy()
                for c, v in md.items():
                    frame[c] = v
                frame["UnitID"] = key + "::" + frame.OriginalUnitID.astype(str)
                # Conservative sensitivity: preserve all rows and flag them.
                duplicate = (
                    root / "reports" / "primary_300s" / key / "PROCESSING_VERIFIED.json"
                )
                excluded = []
                if duplicate.exists():
                    excluded = json.loads(duplicate.read_text())[
                        "dense_MEA_identity_review"
                    ]["review_original_unit_ids"]
                frame["DuplicateReview"] = frame.OriginalUnitID.isin(excluded)
                units.append(frame)
            if (folder / "NetworkFeatures.csv").exists():
                net = pd.read_csv(folder / "NetworkFeatures.csv")
                for c, v in md.items():
                    net[c] = v
                net["QCUnitYield"] = status.get("units", np.nan)
                recordings.append(net)
        expected = len(statuses)
        progress = root / "REPORT_STATUS.json"
        if progress.exists():
            expected = json.loads(progress.read_text()).get("expected", expected)
        return cls(
            pd.concat(units, ignore_index=True) if units else pd.DataFrame(),
            pd.concat(recordings, ignore_index=True) if recordings else pd.DataFrame(),
            dict(
                feature_origin="native DeePhys CSV exports",
                primary_duration_s=300,
                recordings=list(statuses),
                E_I_labels="not supplied",
                expected_recordings=expected,
                genotype_mapping="unconfirmed",
            ),
        )

    def matrix(
        self,
        level="unit",
        *,
        exclude_stage_missingness=True,
        exclude_duplicate_review=True,
    ):
        frame = (self.units if level == "unit" else self.recordings).copy()
        if level == "unit" and exclude_duplicate_review and "DuplicateReview" in frame:
            frame = frame.loc[~frame.DuplicateReview].reset_index(drop=True)
        cols = [
            c
            for c in frame.select_dtypes(include=np.number)
            if c not in METADATA and c != "DuplicateReview"
        ]
        # Parent ACGs use a longer/different observation history than the
        # child window and must be explicitly requested in a separate analysis.
        cols = [c for c in cols if not c.startswith("Parent_")]
        if exclude_stage_missingness:
            # Staged extensions must never identify processing progress.
            cols = [
                c
                for c in cols
                if "STTC" not in c.upper()
                and not c.startswith("SpatialDecayTau")
                and c not in ("DistanceFCCorrelation", "MeanFC_NearNeighbors")
            ]
        X = frame[cols].replace([np.inf, -np.inf], np.nan)
        metadata = frame[[c for c in frame if c in METADATA]].copy()
        return X, metadata

    def trajectories(self, days=(15, 18, 19, 20, 21)):
        """Exact DIV profiles; numbered well identity remains an assumption."""
        frame = self.recordings
        selected = frame[frame.DIV.isin(days)].copy()
        if selected.duplicated(["Replicate", "Well", "DIV"]).any():
            raise ValueError("ambiguous repeated numbered well/day")
        complete = (
            selected.groupby(["Replicate", "Well"])
            .DIV.agg(set)
            .apply(lambda x: set(days) <= x)
        )
        keys = set(complete[complete].index)
        selected = selected[
            [(r, w) in keys for r, w in zip(selected.Replicate, selected.Well)]
        ]
        cols = [
            c for c in selected.select_dtypes(include=np.number) if c not in METADATA
        ]
        return selected.pivot(index=["Replicate", "Well"], columns="DIV", values=cols)
