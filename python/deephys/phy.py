"""Read-only Phy inputs. params.py is parsed as data, never executed."""

from pathlib import Path
from dataclasses import dataclass
import ast
import numpy as np


def read_params(path):
    result = {}
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return result


@dataclass
class PhyRecording:
    path: Path
    sampling_rate: float
    duration: float
    spike_samples: np.ndarray
    spike_clusters: np.ndarray
    templates: np.ndarray
    unit_ids: np.ndarray
    template_unit_ids: np.ndarray
    positions: np.ndarray
    whitening_inverse: np.ndarray | None
    template_gain_uv: float | None

    @classmethod
    def read(cls, path, *, duration=None, sampling_rate=None, template_gain_uv=None):
        path = Path(path).resolve()
        p = read_params(path / "params.py")
        fs = float(sampling_rate or p.get("sample_rate", 0))
        if not fs > 0:
            raise ValueError("sampling rate must be explicit")
        if duration is None:
            if "n_samples" not in p:
                raise ValueError(
                    "duration or n_samples required; last spike is not recording duration"
                )
            duration = p["n_samples"] / fs
        if not (duration > 0 and np.isfinite(duration)):
            raise ValueError("invalid duration")
        load = lambda name: np.load(path / name, mmap_mode="r", allow_pickle=False)
        samples = load("spike_times.npy").reshape(-1)
        clusters = load("spike_clusters.npy").reshape(-1)
        if samples.dtype.kind not in "iu" or len(samples) != len(clusters):
            raise ValueError("invalid spike samples/cluster alignment")
        if len(samples) and (
            samples.min() < 0 or samples.max() >= np.ceil(duration * fs)
        ):
            raise ValueError("out-of-window spike samples")
        if np.any(np.diff(samples.astype(np.int64)) < 0):
            raise ValueError("global spike samples are not sorted")
        templates = load("templates.npy")
        positions = load("channel_positions.npy")
        if templates.ndim != 3 or templates.shape[2] != len(positions):
            raise ValueError(
                "dense template/channel mapping required; expand sparse detector templates explicitly"
            )
        mapped = (
            load("phy_ids.npy").reshape(-1)
            if (path / "phy_ids.npy").exists()
            else np.arange(len(templates))
        )
        ids = np.unique(clusters)
        if (
            len(mapped) != len(templates)
            or len(np.unique(mapped)) != len(mapped)
            or not np.isin(ids, mapped).all()
        ):
            raise ValueError(
                "original cluster to template mapping is incomplete or ambiguous"
            )
        winv = (
            load("whitening_mat_inv.npy")
            if (path / "whitening_mat_inv.npy").exists()
            else None
        )
        if winv is not None and winv.shape != (len(positions), len(positions)):
            raise ValueError("invalid whitening inverse")
        return cls(
            path,
            fs,
            float(duration),
            samples,
            clusters,
            templates,
            ids,
            mapped,
            positions,
            winv,
            template_gain_uv,
        )

    def spike_train(self, unit_id):
        samples = np.asarray(
            self.spike_samples[self.spike_clusters == unit_id], np.int64
        )
        return samples / self.sampling_rate

    def physical_template(self, unit_id):
        if self.template_gain_uv is None:
            raise ValueError(
                "template amplitude calibration in uV must be supplied explicitly"
            )
        row = np.flatnonzero(self.template_unit_ids == unit_id)
        if len(row) != 1:
            raise KeyError(unit_id)
        template = np.asarray(self.templates[row[0]], float)
        if self.whitening_inverse is not None:
            inv = np.asarray(self.whitening_inverse)
            if not np.array_equal(inv, np.eye(len(inv), dtype=inv.dtype)):
                template = template @ inv
        return template * self.template_gain_uv

    def duplicate_counts(self):
        return {
            int(uid): int(
                np.sum(np.diff(self.spike_samples[self.spike_clusters == uid]) == 0)
            )
            for uid in self.unit_ids
        }
