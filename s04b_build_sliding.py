
import numpy as np, pandas as pd
from pathlib import Path
import config
from config import ALL_LOCATIONS, WEATHER_DIR
from s04_extract_weekly import EPW_FEATURES

WIN, STRIDE = 168, 24
RUNS_DIR = Path(config.OUTPUT_DIR)
OUT_DIR  = Path(config.DATASET_DIR)


def read_epw(loc):
    """Parse exactly as s04 does: same pandas call, same column map."""
    path = Path(WEATHER_DIR) / ALL_LOCATIONS[loc]["epw"]
    df = pd.read_csv(path, skiprows=8, header=None)
    a = df.iloc[:8760, list(EPW_FEATURES.values())].to_numpy(dtype=np.float32)
    assert a.shape == (8760, len(EPW_FEATURES)), f"{path}: {a.shape}"
    return a


def energy_csv(loc, d):
    for cand in (RUNS_DIR / loc / f"d{d:03d}" / "eplusout.csv",
                 RUNS_DIR / f"{loc}_d{d:03d}" / "eplusout.csv"):
        if cand.exists():
            return cand
    raise SystemExit(f"no eplusout.csv for {loc} d{d:03d} under {RUNS_DIR} "
                     f"- run: dir \"{RUNS_DIR / loc}\" and paste it")


def hourly_energy_kwh(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if ":Facility" in c]   # Electricity + NaturalGas
    assert cols, f"{csv_path}: no Facility meters found"
    e = np.zeros(len(df), dtype=np.float64)
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64)
        if "[J]" in c:
            v = v / 3.6e6
        e += v
    assert len(e) == 8760, f"{csv_path}: {len(e)} hourly rows"
    return e.astype(np.float32)


def main():
    starts = np.arange(0, 8760 - WIN + 1, STRIDE)      # 359 starts
    for loc in config.LOCATIONS:                       # training stations only
        weekly = np.load(OUT_DIR / f"{loc}.npz", allow_pickle=True)
        B50 = weekly["B"].reshape(50, 52, -1)[:, 0, :] # one design-vector per design
        epw = read_epw(loc)
        Wloc = np.stack([epw[s:s+WIN] for s in starts])        # (359,168,8)

        # self-check against the weekly dataset: every 7th window is a
        # calendar week start, so the first 52 must match W[:52] exactly
        if not np.array_equal(weekly["W"][:52].astype(np.float32),
                              Wloc[::7][:52]):
            raise SystemExit(f"{loc}: sliding weather does not match the "
                             f"weekly dataset - channel maps differ")

        W, B, E, D = [], [], [], []
        for d in range(50):
            e_h = hourly_energy_kwh(energy_csv(loc, d))
            E.append(np.array([e_h[s:s+WIN].sum() for s in starts], np.float32))
            W.append(Wloc)
            B.append(np.repeat(B50[d:d+1], len(starts), axis=0))
            D.append(np.full(len(starts), d, np.int32))
        np.savez_compressed(OUT_DIR / f"{loc}_sliding.npz",
                            W=np.concatenate(W), B=np.concatenate(B),
                            E=np.concatenate(E), design_ids=np.concatenate(D))
        print(f"{loc}: {50*len(starts)} sliding samples written")


if __name__ == "__main__":
    main()