"""Fast end-to-end check on synthetic features (CPU, under a minute). Run: python -m tests.test_smoke"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import load_features
from src.train import main

D, N_TOK, N_REG = 32, 40, 49


def make_part(folder, prefix, splits, rng):
    """Synthetic part whose label leaks weakly into both modalities, so models can learn something."""
    folder.mkdir(parents=True)
    n = len(splits)
    y = (rng.random(n) < 0.3).astype(int)
    direction = np.random.default_rng(1).choice([-1.0, 1.0], D)   # signal along a fixed direction (survives LayerNorm)
    shift = (y[:, None] - 0.3) * 0.8 * direction
    pd.DataFrame({"pair_id": [f"{prefix}_{i:05d}.jpg" for i in range(n)], "question": "what is this?",
                  "official_split": splits, "y": y, "unans_votes": np.clip(y * 7 + rng.integers(0, 3, n), 0, 10)}
                 ).to_csv(folder / "meta.csv", index=False)
    mask = np.zeros((n, N_TOK), np.uint8)
    mask[:, :8] = 1
    arrays = {"img_cls": rng.normal(size=(n, D)) + shift, "img_tok": rng.normal(size=(n, N_REG, D)) + shift[:, None],
              "txt_mean": rng.normal(size=(n, D)) + shift, "txt_tok": rng.normal(size=(n, N_TOK, D)) + shift[:, None],
              "txt_mask": mask}
    for k, a in arrays.items():
        np.save(folder / f"{k}.npy", a.astype(np.uint8 if k == "txt_mask" else np.float16))


def run():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_part(tmp / "features" / "trainval", "tv", ["train"] * 300 + ["val"] * 80, rng)
        make_part(tmp / "features" / "test", "te", ["test"] * 100, rng)
        cfg = yaml.safe_load(open("configs/default.yaml"))
        cfg["paths"] = {"features_dir": str(tmp / "features"), "runs_dir": str(tmp / "runs")}
        cfg["training"].update(epochs=2, batch_size=64, train_probe_size=100)
        (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg))

        main(["--config", str(tmp / "cfg.yaml"), "--model", "all", "--seeds", "0"])

        for name in ["question_only", "image_only", "early", "late", "cross_attention", "gated"]:
            pred = pd.read_csv(tmp / "runs" / name / "seed0" / "predictions.csv")
            assert set(pred.split) == {"dev", "test"}, name
            assert len(pred) == 180 and pred.p_unanswerable.between(0, 1).all(), name
            assert {"pair_id", "split", "model", "seed", "y", "p_unanswerable"} <= set(pred.columns), name
        assert "gate_q" in pd.read_csv(tmp / "runs" / "gated" / "seed0" / "predictions.csv").columns
        summary = json.loads((tmp / "runs" / "gated" / "seed0" / "summary.json").read_text())
        assert summary["params"] > 0 and summary["epochs_run"] >= 1

        soft = load_features(tmp / "features", ["trainval", "test"], label="soft")
        assert soft.meta["target"].between(0, 1).all() and soft.meta["target"].nunique() > 2
        assert {s: len(soft.index(s)) for s in ("train", "dev", "test")} == {"train": 300, "dev": 80, "test": 100}
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    run()
