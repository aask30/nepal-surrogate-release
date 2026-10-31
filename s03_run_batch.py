"""
Step 3 of the pipeline: run EnergyPlus for every (design, location) pair in parallel.

Each run lands in runs/<location>/<design_id>/ with eplusout.csv (hourly meters).
Failed runs are listed at the end; rerun the script and completed runs are skipped.
"""

import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import ENERGYPLUS_EXE, WEATHER_DIR, OUTPUT_DIR, ALL_LOCATIONS


def run_one(idf_path: Path, epw: Path, out_dir: Path) -> tuple[str, bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "eplusout.csv"
    if marker.exists():
        return (out_dir.name, True, "skipped (exists)")
    cmd = [str(ENERGYPLUS_EXE), "-r",          # -r = run ReadVarsESO -> eplusout.csv
           "-w", str(epw), "-d", str(out_dir), str(idf_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        ok = res.returncode == 0 and marker.exists()
        msg = "ok" if ok else (res.stderr or res.stdout)[-400:]
        return (out_dir.name, ok, msg)
    except Exception as exc:                                   # noqa: BLE001
        return (out_dir.name, False, repr(exc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True, choices=list(ALL_LOCATIONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="run a single design_id (smoke test)")
    args = ap.parse_args()

    epw = WEATHER_DIR / ALL_LOCATIONS[args.location]["epw"]
    if not epw.exists():
        raise SystemExit(f"EPW not found: {epw}")

    idf_dir = OUTPUT_DIR / "idfs"
    idfs = sorted(idf_dir.glob("*.idf"))
    if args.only:
        idfs = [p for p in idfs if p.stem == args.only]
    if not idfs:
        raise SystemExit("No IDFs found — run s02_build_idfs.py first.")

    loc_dir = OUTPUT_DIR / args.location
    jobs = [(p, epw, loc_dir / p.stem) for p in idfs]
    print(f"{len(jobs)} simulations -> {loc_dir}  ({args.workers} workers)")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {msg[:120]}")
            if not ok:
                failures.append(name)

    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
        print("Inspect eplusout.err in each failed folder; rerun to retry.")
    else:
        print("\nAll simulations completed.")


if __name__ == "__main__":
    main()