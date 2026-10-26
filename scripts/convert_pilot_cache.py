"""One-off: turn the pilot notebook's cached features into the 'trainval' part used in Project 2.2.

Run once in Colab after mounting Drive (from the repo root):
    python scripts/convert_pilot_cache.py
The pilot files stay where they are, so the pilot notebook still finds its cache.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

FEATURE_FILES = ("img_cls", "img_tok", "txt_mean", "txt_tok", "txt_mask")


def main():
    ap = argparse.ArgumentParser(description="Convert the pilot feature cache into the trainval part")
    ap.add_argument("--pilot-dir", default="/content/drive/MyDrive/vizwiz_fusion/features")
    ap.add_argument("--out", default=None, help="default: <pilot-dir>/trainval")
    ap.add_argument("--fingerprint", default=None, help="only needed if the pilot cache holds several runs")
    args = ap.parse_args()

    pilot = Path(args.pilot_dir)
    out = Path(args.out) if args.out else pilot / "trainval"
    metas = sorted(pilot.glob(f"meta_{args.fingerprint or '*'}.csv"))
    if len(metas) != 1:
        raise SystemExit(f"expected exactly one meta_<fingerprint>.csv in {pilot}, found {[m.name for m in metas]}; "
                         "pass --fingerprint to choose")
    fp = metas[0].stem.split("_", 1)[1]

    out.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_FILES:
        src, dst = pilot / f"{name}_{fp}.npy", out / f"{name}.npy"
        if dst.exists():
            print(f"{dst.name}: already there, skipped")
            continue
        shutil.copyfile(src, dst)
        print(f"{dst.name}: copied ({dst.stat().st_size / 1e9:.2f} GB)")

    meta = pd.read_csv(metas[0]).rename(columns={"image": "pair_id", "split": "pilot_split"})
    meta.to_csv(out / "meta.csv", index=False)
    print(f"meta.csv: {len(meta):,} rows, official splits {meta['official_split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
