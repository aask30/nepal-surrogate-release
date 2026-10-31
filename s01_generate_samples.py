"""
Step 1 of the pipeline: Latin Hypercube Sampling of the 14-parameter design space.
Produces datasets/design_matrix.csv — one row per design variant.
The SAME design matrix is reused for every location so that cross-location
differences come from weather alone
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc

from config import PARAM_RANGES, N_DESIGNS, RANDOM_SEED, DATASET_DIR


def main() -> None:
    names = list(PARAM_RANGES.keys())
    lows = np.array([PARAM_RANGES[n][0] for n in names])
    highs = np.array([PARAM_RANGES[n][1] for n in names])

    sampler = qmc.LatinHypercube(d=len(names), seed=RANDOM_SEED)
    unit = sampler.random(n=N_DESIGNS)
    scaled = qmc.scale(unit, lows, highs)

    df = pd.DataFrame(scaled, columns=names)
    df.insert(0, "design_id", [f"d{i:03d}" for i in range(N_DESIGNS)])

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / "design_matrix.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} designs x {len(names)} parameters -> {out}")
    print(df.describe().loc[["min", "max"]].T.round(3))


if __name__ == "__main__":
    main()
