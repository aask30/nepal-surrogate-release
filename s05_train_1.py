"""
Step 5 of the pipeline: train surrogate models on the (W, B, E) datasets.
Aligned with Manmatharasan et al. (2026), Energy & Buildings 359, 117251.

W : weekly weather sequences (n, 168, 8)
B : design vectors            (n, 14)
E : weekly site energy [kWh]  (n,)

"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATASET_DIR, LOCATIONS, PARAM_RANGES

SEED = 42
VAL_DESIGN_FRAC = 0.2          # custom/grid modes: designs held out for val
torch.manual_seed(SEED)
np.random.seed(SEED)


# data

def load_location(loc):
    d = np.load(DATASET_DIR / f"{loc}.npz", allow_pickle=True)
    return (d["W"].astype(np.float32), d["B"].astype(np.float32),
            d["E"].astype(np.float32), d["design_ids"].astype(str),
            d["week_ids"].astype(int))


def concat_locs(locs):
    Ws, Bs, Es, Ds, Ks, Ls = [], [], [], [], [], []
    for loc in locs:
        W, B, E, d, k = load_location(loc)
        Ws.append(W); Bs.append(B); Es.append(E); Ds.append(d); Ks.append(k)
        Ls.append(np.full(len(E), loc))
    return (np.concatenate(Ws), np.concatenate(Bs), np.concatenate(Es),
            np.concatenate(Ds), np.concatenate(Ks), np.concatenate(Ls))


class Normalizer:
    """Weather: per-channel min-max fitted on TRAIN, no clipping (paper).
    Design: min-max from PARAM_RANGES. Energy: z-score on TRAIN."""

    def fit(self, W, B, E):
        self.w_min = W.min(axis=(0, 1), keepdims=True)
        self.w_rng = W.max(axis=(0, 1), keepdims=True) - self.w_min + 1e-6
        lows = np.array([v[0] for v in PARAM_RANGES.values()], np.float32)
        highs = np.array([v[1] for v in PARAM_RANGES.values()], np.float32)
        self.b_low, self.b_rng = lows, (highs - lows)
        self.e_mean, self.e_std = float(E.mean()), float(E.std() + 1e-6)
        return self

    def tf(self, W, B, E=None):
        Wn = (W - self.w_min) / self.w_rng          # may exceed [0,1]: ok
        Bn = (B - self.b_low) / self.b_rng
        if E is None:
            return Wn, Bn
        return Wn, Bn, (E - self.e_mean) / self.e_std

    def inv_e(self, En):
        return En * self.e_std + self.e_mean

    def state(self):
        return {k: getattr(self, k) for k in
                ("w_min", "w_rng", "b_low", "b_rng", "e_mean", "e_std")}


# models

class DesignHead(nn.Module):
    def __init__(self, w_dim, b_dim=14, hidden=128):
        super().__init__()
        self.b_net = nn.Sequential(nn.Linear(b_dim, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(w_dim + 64, hidden), nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, w_emb, b):
        return self.head(torch.cat([w_emb, self.b_net(b)], dim=1)).squeeze(1)


class MLPSurrogate(nn.Module):
    def __init__(self, n_ch=8):
        super().__init__()
        self.w_net = nn.Sequential(nn.Linear(4 * n_ch, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.fuse = DesignHead(64)

    def forward(self, W, B):
        stats = torch.cat([W.mean(1), W.std(1), W.amin(1), W.amax(1)], dim=1)
        return self.fuse(self.w_net(stats), B)


class TCNSurrogate(nn.Module):
    def __init__(self, n_ch=8, width=64):
        super().__init__()
        layers, c_in = [], n_ch
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv1d(c_in, width, 3, padding=dilation,
                                 dilation=dilation),
                       nn.ReLU(), nn.BatchNorm1d(width)]
            c_in = width
        self.tcn = nn.Sequential(*layers)
        self.fuse = DesignHead(width)

    def forward(self, W, B):
        h = self.tcn(W.permute(0, 2, 1))
        return self.fuse(h.mean(dim=2), B)


class TransformerSurrogate(nn.Module):
    def __init__(self, n_ch=8, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(n_ch, d_model)
        pos = torch.zeros(168, d_model)
        t = torch.arange(168).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(t * div)
        pos[:, 1::2] = torch.cos(t * div)
        self.register_buffer("pos", pos.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 128, 0.1,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.fuse = DesignHead(d_model)

    def forward(self, W, B):
        h = self.encoder(self.proj(W) + self.pos)
        return self.fuse(h.mean(dim=1), B)


class ConvAutoencoder(nn.Module):
    """Approach 2: unsupervised weekly-weather autoencoder (paper 3.5)."""

    def __init__(self, n_ch=8, base=32, embed=56):
        super().__init__()
        self.embed = embed

        def block(ci, co):
            return [nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co),
                    nn.ReLU(),
                    nn.Conv1d(co, co, 3, padding=1), nn.BatchNorm1d(co),
                    nn.ReLU(), nn.MaxPool1d(2)]
        self.enc = nn.Sequential(*block(n_ch, base), *block(base, 2 * base),
                                 *block(2 * base, 4 * base))   # 168 -> 21
        self.to_embed = nn.Sequential(nn.Conv1d(4 * base, embed, 1),
                                      nn.AdaptiveAvgPool1d(1))
        self.from_embed = nn.Linear(embed, 4 * base * 21)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(4 * base, 2 * base, 3,
                                                   padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(2 * base, base, 3,
                                                   padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(base, n_ch, 3, padding=1))
        self.base = base

    def encode(self, W):                       # W: (n, 168, ch)
        h = self.enc(W.permute(0, 2, 1))
        return self.to_embed(h).squeeze(2)     # (n, embed)

    def forward(self, W):
        z = self.encode(W)
        h = self.from_embed(z).view(-1, 4 * self.base, 21)
        return self.dec(h).permute(0, 2, 1)    # (n, 168, ch)


class AESurrogate(nn.Module):
    """Frozen pre-trained encoder + FFNN predictor (only head trains)."""

    def __init__(self, ae):
        super().__init__()
        self.ae = ae
        for p in self.ae.parameters():
            p.requires_grad = False
        self.fuse = DesignHead(ae.embed)

    def forward(self, W, B):
        with torch.no_grad():
            z = self.ae.encode(W)
        return self.fuse(z, B)

    def trainable_parameters(self):
        return self.fuse.parameters()


# metrics (paper Sec. 3.6)

def smape(y, p):
    return 100.0 * float(np.mean(np.abs(y - p)
                                 / ((np.abs(y) + np.abs(p)) / 2 + 1e-9)))


def full_metrics(y, p, d_ids):
    err = p - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean = float(np.mean(y))
    # annual aggregation + per-design Pearson over 52-week profiles
    ya, pa, rs = [], [], []
    for d in np.unique(d_ids):
        m = d_ids == d
        ya.append(y[m].sum()); pa.append(p[m].sum())
        if y[m].std() > 1e-9 and p[m].std() > 1e-9:
            rs.append(float(np.corrcoef(y[m], p[m])[0, 1]))
    return {"SMAPE_weekly_%": smape(y, p),
            "SMAPE_annual_%": smape(np.array(ya), np.array(pa)),
            "Pearson_r_mean": float(np.mean(rs)) if rs else float("nan"),
            "RMSE_kWh": rmse,
            "CV-RMSE_%": 100.0 * rmse / mean,
            "NMBE_%": 100.0 * float(np.mean(err)) / mean,
            "R2": 1.0 - float(np.sum(err ** 2))
                      / float(np.sum((y - mean) ** 2))}


# training machinery

def make_loader(norm, W, B, E, batch, shuffle):
    Wn, Bn, En = norm.tf(W, B, E)
    ds = TensorDataset(torch.from_numpy(Wn), torch.from_numpy(Bn),
                       torch.from_numpy(En))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def pretrain_autoencoder(norm, ae_locs, device, epochs, batch=128, lr=1e-3):
    """Unsupervised on UNIQUE weekly weather (52 weeks/location)."""
    weeks = [load_location(loc)[0][:52] for loc in ae_locs]
    W = np.concatenate(weeks)
    Wn, _ = norm.tf(W, np.zeros((len(W), 14), np.float32))
    ae = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.from_numpy(Wn)),
                        batch_size=batch, shuffle=True)
    loss_fn = nn.MSELoss()
    print(f"pretraining autoencoder on {len(W)} unique weeks "
          f"from {len(ae_locs)} locations")
    for ep in range(1, epochs + 1):
        ae.train(); tot = 0.0
        for (w,) in loader:
            w = w.to(device)
            opt.zero_grad()
            loss = loss_fn(ae(w), w)
            loss.backward(); opt.step()
            tot += loss.item() * len(w)
        if ep % 10 == 0 or ep == 1:
            print(f"  AE epoch {ep:3d}  recon MSE {tot / len(W):.4f}")
    ae.eval()
    return ae


def build_model(name, norm, device, args, ae_locs):
    if name == "mlp":
        return MLPSurrogate().to(device)
    if name == "tcn":
        return TCNSurrogate().to(device)
    if name == "transformer":
        return TransformerSurrogate().to(device)
    if name == "autoencoder":
        ae = pretrain_autoencoder(norm, ae_locs, device, args.ae_epochs)
        return AESurrogate(ae).to(device)
    raise ValueError(name)


def train_model(model, tr_loader, va_loader, device, args, tag=""):
    params = (model.trainable_parameters()
              if isinstance(model, AESurrogate) else model.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=5)
    loss_fn = nn.MSELoss()
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tr, n = 0.0, 0
        for W, B, E in tr_loader:
            W, B, E = W.to(device), B.to(device), E.to(device)
            opt.zero_grad()
            loss = loss_fn(model(W, B), E)
            loss.backward(); opt.step()
            tr += loss.item() * len(E); n += len(E)
        model.eval(); va, m = 0.0, 0
        with torch.no_grad():
            for W, B, E in va_loader:
                W, B, E = W.to(device), B.to(device), E.to(device)
                va += loss_fn(model(W, B), E).item() * len(E); m += len(E)
        tr, va = tr / n, va / m
        sched.step(va)
        hist.append({"epoch": ep, "train_mse": tr, "val_mse": va})
        mark = ""
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = "  <- best"
        else:
            bad += 1
        if not args.quiet:
            print(f"{tag}epoch {ep:3d}  train {tr:.4f}  val {va:.4f}{mark}")
        if bad >= args.patience:
            if not args.quiet:
                print(f"{tag}early stop")
            break
    print(f"{tag}trained in {time.time() - t0:.0f} s "
          f"(best val MSE {best:.4f})")
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def predict(model, norm, W, B, device, batch=512):
    Wn, Bn = norm.tf(W, B)
    out = []
    with torch.no_grad():
        for i in range(0, len(Wn), batch):
            w = torch.from_numpy(Wn[i:i + batch]).to(device)
            b = torch.from_numpy(Bn[i:i + batch]).to(device)
            out.append(model(w, b).cpu().numpy())
    return norm.inv_e(np.concatenate(out))


def save_profile_plot(out_dir, name, loc, d_ids, w_ids, y, p):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sel = d_ids == d_ids[0]
    order = np.argsort(w_ids[sel])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(w_ids[sel][order], y[sel][order], label="EnergyPlus", lw=2)
    ax.plot(w_ids[sel][order], p[sel][order], "--", label="surrogate", lw=2)
    ax.set_xlabel("week"); ax.set_ylabel("weekly energy [kWh]")
    ax.set_title(f"{name} | {loc} | design {d_ids[0]}")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / f"profile_{loc}.png", dpi=150)
    plt.close(fig)



# modes

def split_by_designs(locs, val_designs):
    W, B, E, d, k, L = concat_locs(locs)
    vmask = np.isin(d, val_designs)
    tr = (W[~vmask], B[~vmask], E[~vmask], d[~vmask], k[~vmask])
    va = (W[vmask], B[vmask], E[vmask], d[vmask], k[vmask])
    return tr, va


def run_training_set(train_locs, args, device, out_dir, grid_row=False):
    """Train on train_locs (val = held-out designs), evaluate everywhere."""
    all_designs = np.unique(load_location(train_locs[0])[3])
    rng = np.random.default_rng(SEED)
    n_val = max(1, int(round(VAL_DESIGN_FRAC * len(all_designs))))
    val_designs = rng.choice(all_designs, size=n_val, replace=False)

    tr, va = split_by_designs(train_locs, val_designs)
    norm = Normalizer().fit(*tr[:3])
    ae_locs = (list(LOCATIONS) if args.ae_weather == "all" else train_locs)
    model = build_model(args.model, norm, device, args, ae_locs)
    tr_loader = make_loader(norm, *tr[:3], args.batch, True)
    va_loader = make_loader(norm, *va[:3], args.batch, False)
    tag = f"[{'+'.join(train_locs)}] " if grid_row else ""
    model, hist = train_model(model, tr_loader, va_loader, device, args, tag)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

    rows = {}
    for loc in LOCATIONS:
        W, B, E, d, k = load_location(loc)
        if loc in train_locs:     # same-location eval: held-out designs only
            m = np.isin(d, val_designs)
            W, B, E, d, k = W[m], B[m], E[m], d[m], k[m]
        p = predict(model, norm, W, B, device)
        rows[loc] = full_metrics(E, p, d)
        save_profile_plot(out_dir, args.model, loc, d, k, E, p)
    df = pd.DataFrame(rows).T
    df.index.name = "test_location"
    df.to_csv(out_dir / "metrics_by_location.csv")
    print(df.round(2).to_string())
    return df


def mode_holdout(args, device):
    test, val = args.test_location, args.val_location
    train_locs = [l for l in LOCATIONS if l not in (test, val)]
    print(f"train: {train_locs}\nval: {val}   test: {test}")
    trW, trB, trE, trd, trk, _ = concat_locs(train_locs)
    norm = Normalizer().fit(trW, trB, trE)
    ae_locs = (list(LOCATIONS) if args.ae_weather == "all" else train_locs)
    model = build_model(args.model, norm, device, args, ae_locs)
    vaW, vaB, vaE, _, _, _ = concat_locs([val])
    model, hist = train_model(
        model, make_loader(norm, trW, trB, trE, args.batch, True),
        make_loader(norm, vaW, vaB, vaE, args.batch, False), device, args)

    W, B, E, d, k, _ = concat_locs([test])
    p = predict(model, norm, W, B, device)
    m = full_metrics(E, p, d)
    out_dir = Path("results") / f"{args.model}_test-{test}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"model={args.model}  test={test}  val={val}",
             *[f"{k_}: {v:.3f}" for k_, v in m.items()]]
    print("\n".join(lines))
    (out_dir / "metrics.txt").write_text("\n".join(lines))
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame({"design_id": d, "week": k, "actual_kWh": E,
                  "pred_kWh": p}).to_csv(out_dir / "predictions.csv",
                                         index=False)
    torch.save({"model": args.model, "state_dict": model.state_dict(),
                "normalizer": norm.state()}, out_dir / "model.pt")
    save_profile_plot(out_dir, args.model, test, d, k, E, p)
    print(f"results -> {out_dir}")


def mode_custom(args, device):
    train_locs = [s.strip() for s in args.train_locations.split(",")]
    bad = [l for l in train_locs if l not in LOCATIONS]
    if bad:
        raise SystemExit(f"unknown locations: {bad}")
    out_dir = Path("results") / f"{args.model}_train-{'+'.join(train_locs)}"
    run_training_set(train_locs, args, device, out_dir)
    print(f"results -> {out_dir}")


def mode_grid(args, device):
    out_root = Path("results") / f"grid_{args.model}"
    heat = {k: pd.DataFrame(index=list(LOCATIONS),
                            columns=list(LOCATIONS), dtype=float)
            for k in ("SMAPE_weekly_%", "SMAPE_annual_%", "Pearson_r_mean")}
    for train_loc in LOCATIONS:
        df = run_training_set([train_loc], args, device,
                              out_root / f"train-{train_loc}", grid_row=True)
        for k in heat:
            heat[k].loc[train_loc] = df[k]
    for k, hm in heat.items():
        fname = out_root / f"heatmap_{k.replace('%', 'pct').replace('/', '')}.csv"
        hm.index.name = "train\\test"
        hm.to_csv(fname)
        print(f"\n=== {k} (rows=train location, cols=test location) ===")
        print(hm.astype(float).round(2).to_string())
    print(f"\nheatmaps -> {out_root}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn",
                    choices=["mlp", "tcn", "transformer", "autoencoder"])
    ap.add_argument("--test-location", choices=list(LOCATIONS))
    ap.add_argument("--val-location", choices=list(LOCATIONS))
    ap.add_argument("--train-locations",
                    help="comma-separated training locations (paper mode)")
    ap.add_argument("--grid", action="store_true",
                    help="single-location 10x10 heatmap replication")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--ae-epochs", type=int, default=40,
                    help="autoencoder pretraining epochs")
    ap.add_argument("--ae-weather", choices=["train", "all"], default="all",
                    help="weather used for AE pretraining (paper: all; "
                         "weather files are simulation-free)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-epoch logs (useful with --grid)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.grid:
        if not args.quiet:
            args.quiet = True   # grid is 10 trainings; keep the log readable
        mode_grid(args, device)
    elif args.train_locations:
        mode_custom(args, device)
    elif args.test_location and args.val_location:
        if args.test_location == args.val_location:
            raise SystemExit("val and test locations must differ")
        mode_holdout(args, device)
    else:
        raise SystemExit("choose a mode: --grid | --train-locations ... | "
                         "--test-location ... --val-location ...")


if __name__ == "__main__":
    main()
