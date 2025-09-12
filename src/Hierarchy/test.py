#!/usr/bin/env python3
"""
Collect shape[0] sizes for every .npy file in a directory, sort them
(from largest to smallest), and save the results as a CSV.

Usage:  python make_file_sizes.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# ---- paths ----------------------------------------------------------
SRC_DIR   = Path("/home/lvcardoso/recurrent-transformer/video_feature/rt_anet_feat/trainval")
DEST_CSV  = Path("/home/bernardop/file_sizes.csv")

# ---- gather sizes ---------------------------------------------------
records = []
for npy_path in SRC_DIR.glob("*.npy"):
    try:
        arr = np.load(npy_path, mmap_mode="r")  # header-only read
        records.append((npy_path.name, int(arr.shape[0])))
    except Exception as exc:
        print(f"⚠️  Skipping {npy_path.name}: {exc}")

# ---- build & save table ---------------------------------------------
df = pd.DataFrame(records, columns=["file_name", "shape0"]) \
       .sort_values("shape0", ascending=False)

DEST_CSV.parent.mkdir(parents=True, exist_ok=True)  # ensure target dir exists
df.to_csv(DEST_CSV, index=False)
print(f"✅  Wrote {len(df)} entries to {DEST_CSV}")
