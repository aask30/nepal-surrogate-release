from pathlib import Path
import pandas as pd
from config import OUTPUT_DIR, ALL_LOCATIONS

bad = []
checked = 0
for loc in ALL_LOCATIONS:
    loc_dir = OUTPUT_DIR / loc
    if not loc_dir.exists():
        print(f"(skipping {loc}: not simulated yet)")
        continue
    for i in range(50):
        d = f"d{i:03d}"
        csv = loc_dir / d / "eplusout.csv"
        checked += 1
        if not csv.exists():
            bad.append((loc, d, "missing csv")); continue
        try:
            n = len(pd.read_csv(csv, usecols=[0]))
            if n < 8760:
                bad.append((loc, d, f"only {n} rows"))
        except Exception as e:
            bad.append((loc, d, f"unreadable: {e}"))

if not bad:
    print(f"ALL {checked} CHECKED RUNS COMPLETE AND FULL-LENGTH")
else:
    print(f"{len(bad)} bad runs:")
    for loc, d, why in bad:
        print(f"  {loc}/{d}: {why}")
    print("\nTo fix, delete these folders then rerun the batch loop:")
    for loc, d, _ in bad:
        print(f'rmdir /s /q "runs\\{loc}\\{d}"')