"""
Step 6 of the pipeline: ZERO-SHOT evaluation of a saved surrogate on any
location — Nepal stations or Asian EVAL_LOCATIONS cities — with no
training. Requires the location's .npz from s04 and a model.pt saved by


"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ALL_LOCATIONS
from s05_train_2 import (MLPSurrogate, TCNSurrogate, TransformerSurrogate,
                         ConvAutoencoder, AESurrogate, Normalizer,
                         load_location, full_metrics, predict,
                         save_profile_plot)


def bare_model(name):
    if name == "mlp":
        return MLPSurrogate()
    if name == "tcn":
        return TCNSurrogate()
    if name == "transformer":
        return TransformerSurrogate()
    if name == "autoencoder":
        return AESurrogate(ConvAutoencoder())
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="results folder containing model.pt")
    ap.add_argument("--locations", required=True,
                    help="comma-separated locations to evaluate")
    args = ap.parse_args()

    locs = [s.strip() for s in args.locations.split(",")]
    bad = [l for l in locs if l not in ALL_LOCATIONS]
    if bad:
        raise SystemExit(f"unknown locations: {bad} — add them to "
                         f"config.EVAL_LOCATIONS and run s03/s04 first")
    missing = [l for l in locs
               if not (Path("datasets") / f"{l}.npz").exists()]
    if missing:
        raise SystemExit(f"no dataset for: {missing} — run "
                         f"s03_run_batch.py and s04_extract_weekly.py first")

    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu",
                      weights_only=False)
    model = bare_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = Normalizer()
    for key, val in ckpt["normalizer"].items():
        setattr(norm, key, val)
    device = torch.device("cpu")

    print(f"model: {ckpt['model']}  trained on: {ckpt.get('train_locs')}  "
          f"scaler: {ckpt.get('weather_scaler', 'unknown')}")
    rows = {}
    for loc in locs:
        W, B, E, d, k = load_location(loc)
        p = predict(model, norm, W, B, device)
        rows[loc] = full_metrics(E, p, d)
        pd.DataFrame({"design_id": d, "week": k, "actual_kWh": E,
                      "pred_kWh": p}).to_csv(
            model_dir / f"predictions_{loc}.csv", index=False)
        save_profile_plot(model_dir, ckpt["model"], loc, d, k, E, p)

    df = pd.DataFrame(rows).T
    df.index.name = "test_location"
    df.to_csv(model_dir / "zeroshot_metrics.csv")
    print(df.round(2).to_string())
    print(f"results -> {model_dir / 'zeroshot_metrics.csv'}")


if __name__ == "__main__":
    main()