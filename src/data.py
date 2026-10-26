"""Load cached features and assign the Project 2.2 splits.

Each feature "part" is a folder holding meta.csv plus one .npy file per array, with rows in the same
order (format in docs/data_contract.md). Splits follow the official VizWiz partition:
official train -> train, official validation -> dev (tuning), official test -> test.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FEATURE_FILES = ("img_cls", "img_tok", "txt_mean", "txt_tok", "txt_mask")
REQUIRED_META = ("pair_id", "question", "official_split", "y", "unans_votes")
SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}
N_CROWD_ANSWERS = 10


@dataclass
class FeatureStore:
    """All rows of all parts in memory: meta (one row per pair) and aligned feature tensors."""
    meta: pd.DataFrame
    arrays: dict

    def index(self, split):
        """Row positions of one split ('train', 'dev' or 'test')."""
        return np.where(self.meta["split"].values == split)[0]

    @property
    def dims(self):
        return {"d_q": int(self.arrays["txt_mean"].shape[1]), "d_v": int(self.arrays["img_cls"].shape[1])}

    def batch(self, idx, needs, device):
        """Build the inputs a model asks for (its `needs`), moved to `device` as float32."""
        idx = torch.as_tensor(np.asarray(idx))
        sources = {
            "q_vec": lambda: self.arrays["txt_mean"][idx].to(device).float(),
            "v_vec": lambda: self.arrays["img_cls"][idx].to(device).float(),
            "q_tok": lambda: self.arrays["txt_tok"][idx].to(device).float(),
            "q_mask": lambda: self.arrays["txt_mask"][idx].to(device).bool(),
            "v_tok": lambda: self.arrays["img_tok"][idx].to(device).float(),
        }
        return {k: sources[k]() for k in needs}


def load_features(features_dir, parts, label="hard"):
    """Read and stack every part listed in the config, then check that rows and splits are consistent."""
    metas, arrays = [], {k: [] for k in FEATURE_FILES}
    for part in parts:
        folder = Path(features_dir) / part
        meta = pd.read_csv(folder / "meta.csv")
        missing = [c for c in REQUIRED_META if c not in meta.columns]
        if missing:
            raise ValueError(f"{folder}/meta.csv is missing columns: {missing}")
        for name in FEATURE_FILES:
            arr = np.load(folder / f"{name}.npy", mmap_mode="r")
            if len(arr) != len(meta):
                raise ValueError(f"{folder}/{name}.npy has {len(arr)} rows but meta.csv has {len(meta)}")
            arrays[name].append(arr)
        metas.append(meta.assign(part=part))

    meta = pd.concat(metas, ignore_index=True)
    meta["split"] = meta["official_split"].map(SPLIT_MAP)
    if meta["split"].isna().any():
        bad = sorted(meta.loc[meta["split"].isna(), "official_split"].unique())
        raise ValueError(f"unknown official_split values: {bad}")
    if not meta["pair_id"].is_unique:
        raise ValueError("pair_id must be unique across all parts")

    meta["y"] = meta["y"].astype(np.float32)
    if label == "hard":
        meta["target"] = meta["y"]
    elif label == "soft":
        # soft label: share of the crowd answers that say "unanswerable" (mitigation for label ambiguity)
        meta["target"] = (meta["unans_votes"] / N_CROWD_ANSWERS).clip(0, 1).astype(np.float32)
    else:
        raise ValueError("data.label must be 'hard' or 'soft'")

    tensors = {k: torch.from_numpy(np.ascontiguousarray(np.concatenate(v))) for k, v in arrays.items()}
    return FeatureStore(meta=meta, arrays=tensors)
