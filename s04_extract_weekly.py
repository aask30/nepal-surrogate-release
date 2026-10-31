"""
Step 4: hourly meters -> weekly site energy, joined with designs
and weekly weather tensors. One compressed .npz per location.
"""

import argparse

import numpy as np
import pandas as pd

from config import (OUTPUT_DIR, DATASET_DIR, WEATHER_DIR, ALL_LOCATIONS,
                    PARAM_RANGES, WEEKS_PER_YEAR)

J_PER_KWH = 3.6e6
HOURS = WEEKS_PER_YEAR * 168          # 8736

# EPW columns used as weather features (index per EPW spec, 0-based)
# NOTE: keys and ORDER must match config.WEATHER_BOUNDS.
EPW_FEATURES = {
    "drybulb":   6,
    "dewpoint":  7,
    "rel_hum":   8,
    "pressure":  9,
    "glob_rad":  13,
    "dir_rad":   14,
    "dif_rad":   15,
    "wind_spd":  21,
}


def weekly_energy(csv_path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    meters = [c for c in df.columns
              if "Electricity:Facility" in c or "NaturalGas:Facility" in c]
    if not meters:
        raise ValueError(f"No facility meters in {csv_path}")
    hourly_j = df[meters].sum(axis=1).to_numpy()
    if len(hourly_j) < HOURS:
        raise ValueError(f"{csv_path}: only {len(hourly_j)} hourly rows")
    hourly_kwh = hourly_j[:HOURS] / J_PER_KWH
    return hourly_kwh.reshape(WEEKS_PER_YEAR, 168).sum(axis=1)


def weekly_weather(epw_path) -> np.ndarray:
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    feats = df.iloc[:HOURS, list(EPW_FEATURES.values())].to_numpy(dtype=float)
    return feats.reshape(WEEKS_PER_YEAR, 168, len(EPW_FEATURES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True, choices=list(ALL_LOCATIONS))
    args = ap.parse_args()

    designs = pd.read_csv(DATASET_DIR / "design_matrix.csv")
    param_cols = list(PARAM_RANGES.keys())
    W_weekly = weekly_weather(WEATHER_DIR / ALL_LOCATIONS[args.location]["epw"])

    W, B, E, d_ids, w_ids = [], [], [], [], []
    missing = []
    for row in designs.itertuples(index=False):
        csv_path = OUTPUT_DIR / args.location / row.design_id / "eplusout.csv"
        if not csv_path.exists():
            missing.append(row.design_id)
            continue
        e_wk = weekly_energy(csv_path)
        b_vec = np.array([getattr(row, c) for c in param_cols], dtype=float)
        for wk in range(WEEKS_PER_YEAR):
            W.append(W_weekly[wk])
            B.append(b_vec)
            E.append(e_wk[wk])
            d_ids.append(row.design_id)
            w_ids.append(wk)

    if missing:
        print(f"WARNING: missing simulations for {missing}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASET_DIR / f"{args.location}.npz"
    np.savez_compressed(
        out,
        W=np.stack(W).astype(np.float32),
        B=np.stack(B).astype(np.float32),
        E=np.array(E, dtype=np.float32),
        design_ids=np.array(d_ids),
        week_ids=np.array(w_ids),
        weather_features=np.array(list(EPW_FEATURES.keys())),
        design_params=np.array(param_cols),
    )
    print(f"Wrote {out}:  W{np.stack(W).shape}  B{np.stack(B).shape}  "
          f"E({len(E)},)  from {designs.shape[0] - len(missing)} designs")
    print(f"Weekly energy range: {min(E):.1f} - {max(E):.1f} kWh")


if __name__ == "__main__":
    main()