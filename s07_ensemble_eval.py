"""
s07_ensemble_eval.py - average predictions of K seed checkpoints and
report ensemble metrics + member spread per city.

python s07_ensemble_eval.py --model-dirs dirA,dirB,dirC --locations patna,...
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from s05_train_2 import (TransformerSurrogate, TCNSurrogate, MLPSurrogate,
                         Normalizer, load_location, full_metrics, predict)


def load_member(model_dir, device):
    ck = torch.load(Path(model_dir) / "model.pt", map_location=device,
                    weights_only=False)
    name = ck["model"]
    if name == "transformer":
        model = TransformerSurrogate()
    elif name == "tcn":
        model = TCNSurrogate()
    elif name == "mlp":
        model = MLPSurrogate()
    else:
        raise SystemExit(f"{model_dir}: ensemble of '{name}' not supported here")
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    norm = Normalizer()
    for k, v in ck["normalizer"].items():
        setattr(norm, k, v)
    return model, norm, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", required=True,
                    help="comma-separated results folders containing model.pt")
    ap.add_argument("--locations", required=True,
                    help="comma-separated locations to evaluate")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(d.strip(), device)
               for d in args.model_dirs.split(",")]
    ck0 = members[0][2]
    print(f"ensemble of {len(members)} x {ck0['model']} "
          f"trained on {ck0['train_locs']}  scaler: {ck0['weather_scaler']}")

    rows, out_root = {}, Path(args.model_dirs.split(",")[0].strip()).parent
    for loc in [s.strip() for s in args.locations.split(",")]:
        W, B, E, d, k = load_location(loc)
        preds = np.stack([predict(m, n, W, B, device)
                          for m, n, _ in members])          # (K, n) kWh
        P = preds.mean(axis=0)
        rows[loc] = full_metrics(E, P, d)
        rows[loc]["member_std_kWh"] = float(preds.std(axis=0).mean())
        rows[loc]["member_std_%"] = 100.0 * float(
            (preds.std(axis=0) / np.maximum(np.abs(P), 1e-9)).mean())
    df = pd.DataFrame(rows).T
    df.index.name = "test_location"
    out = out_root / "ensemble_zeroshot_metrics.csv"
    df.to_csv(out)
    print(df.round(2).to_string())
    print(f"results -> {out}")


if __name__ == "__main__":
    main()