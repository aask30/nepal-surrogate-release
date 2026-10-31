import numpy as np, pandas as pd
from pathlib import Path
import config
from s04b_build_sliding import energy_csv

loc = "jumla"
df = pd.read_csv(energy_csv(loc, 0))
print("Facility columns:")
fac = [c for c in df.columns if ":Facility" in c]
for c in fac: print("  ", c)

w = np.load(Path(config.DATASET_DIR) / f"{loc}.npz")
E_npz = w["E"].reshape(50, 52)[0]          # design 0, 52 weeks (assumes design-major order)
print("\nnpz  weeks 0-3 :", np.round(E_npz[:4], 1))

def weekly(col):
    e = df[col].to_numpy(float)
    if "[J]" in col: e = e / 3.6e6
    return e[:8736].reshape(52, 168).sum(1)

for c in fac:
    print(f"{c:45s} weeks 0-3:", np.round(weekly(c)[:4], 1))

if len(fac) > 1:
    tot = sum(weekly(c) for c in fac)
    print("SUM of all facility meters      weeks 0-3:", np.round(tot[:4], 1))