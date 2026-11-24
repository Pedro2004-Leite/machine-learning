"""PyTorch MLP ensemble for Task 1 (HAB), verified and tuned with honest (nested) validation.

The network reproduces the sklearn MLPClassifier that worked best in the exploration, written explicitly in PyTorch:
Glorot-uniform initialisation, ReLU hidden layers, sigmoid output, Adam (lr 1e-3), mini-batches of 200, the L2 penalty
0.5 * alpha * ||W||^2 / batch_size, and the same stopping rule (training loss stops improving by 1e-4 for 10 epochs).
Dropout is an extra option. Each prediction is the mean probability of K networks with different seeds.

Features are the ones of the final Logistic Regression (utils.DEFAULT_SPEC: 2 harmonics of the day, SST, UI, log CHL,
uncertainty, one-hot cluster), z-scored with the training rows only. Only complete rows are used, as in the test.

Evaluation (2013-2019 only, 2020 is scored once at the end as context):
    leave-one-year-out   each year is left out; size/alpha/dropout chosen by an inner leave-one-year-out (nested)
    3-year blocks        2013-15, ..., 2017-19 are left out together and scored with one pooled AUC, which mimics the
                         real test (three new years, one AUC); the configuration is again chosen in the inner folds
Both are compared with the final Logistic Regression (C=5) on exactly the same splits.

Verification (run first): same seed gives identical predictions, one row at a time equals the batch, the PyTorch
network matches sklearn's MLPClassifier on the 3-year blocks, and every fit asserts that no validation year is in
the training rows. PyTorch is NOT allowed in the submission, so nothing is written to best_model/.

    python training_code/torch_mlp.py            full run (parallel)
    python training_code/torch_mlp.py --quick    smoke test
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "best_model"))
import utils  # noqa: E402

DATA = HERE.parents[1] / "data" / "classification_task1_train.csv"
OUT_JSON, OUT_MODEL = HERE / "torch_mlp_results.json", HERE / "torch_mlp_model.pt"
SPEC = utils.DEFAULT_SPEC
YEARS = list(range(2013, 2020))
BLOCKS = [tuple(range(a, a + 3)) for a in range(2013, 2018)]
LEARNING_RATE, BATCH, MAX_EPOCHS, TOL, NO_CHANGE = 1e-3, 200, 2000, 1e-4, 10
GRID = [{"hidden": h, "alpha": a, "dropout": d}
        for h in [(8,), (16,), (32,), (64,), (16, 16)] for a in (0.3, 1.0, 3.0, 10.0) for d in (0.0, 0.2)]
COMP = None   # complete rows of 2013-2020, loaded once per process


def configure_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load():
    raw = pd.read_csv(DATA, parse_dates=["Date"])
    return raw[raw["CHL"].notna()].reset_index(drop=True)


def rows(years):
    return COMP[COMP["Year"].isin(years)]


# ---------------------------------------------------------------- network

class FeedforwardNetwork(nn.Module):
    def __init__(self, n_features, hidden, dropout):
        super().__init__()
        dims = [n_features, *hidden, 1]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        for layer in self.layers:                       # Glorot uniform, as sklearn (factor 6 for ReLU networks)
            bound = np.sqrt(6.0 / (layer.in_features + layer.out_features))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(self.activation(layer(x)))
        return self.layers[-1](x).squeeze(-1)


def train_batch(X, y, model, optimizer, alpha):
    """One Adam step on the mean BCE plus sklearn's L2 penalty 0.5 * alpha * ||W||^2 / batch size."""
    optimizer.zero_grad()
    loss = nn.functional.binary_cross_entropy_with_logits(model(X), y)
    loss = loss + 0.5 * alpha * sum((layer.weight ** 2).sum() for layer in model.layers) / len(X)
    loss.backward()
    optimizer.step()
    return loss.item()


def fit_net(X, y, cfg, seed):
    """Train until the training loss stops improving (sklearn's rule). Returns the model in eval mode and the epochs."""
    configure_seed(seed)
    model = FeedforwardNetwork(X.shape[1], cfg["hidden"], cfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    n, size = len(X), min(BATCH, len(X))
    generator = torch.Generator().manual_seed(seed)
    best, waited = np.inf, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order, total = torch.randperm(n, generator=generator), 0.0
        for i in range(0, n, size):
            idx = order[i:i + size]
            total += train_batch(X[idx], y[idx], model, optimizer, cfg["alpha"]) * len(idx)
        loss = total / n
        waited = waited + 1 if loss > best - TOL else 0
        best = min(best, loss)
        if waited > NO_CHANGE:
            break
    return model.eval(), epoch


def fit_ensemble(train, cfg, seeds):
    state = utils.fit_state(train, SPEC)                        # z-score and clusters from the training rows only
    X = torch.tensor(utils.transform(train, state), dtype=torch.float32)
    y = torch.tensor(train["HAB"].to_numpy(), dtype=torch.float32)
    nets = [fit_net(X, y, cfg, s) for s in seeds]
    return {"state": state, "cfg": cfg, "nets": [m for m, _ in nets], "epochs": [e for _, e in nets]}


@torch.no_grad()
def predict(ensemble, df):
    X = torch.tensor(utils.transform(df, ensemble["state"]), dtype=torch.float32)
    return torch.stack([torch.sigmoid(m(X)) for m in ensemble["nets"]]).mean(0).numpy()


# ---------------------------------------------------------------- jobs (one process each)

def init_worker():
    global COMP
    torch.set_num_threads(1)
    COMP = load()


def job(args):
    """Fit an ensemble on train_years and return the mean probability for the rows of val_years."""
    train_years, val_years, cfg, seeds = args
    assert not set(train_years) & set(val_years), "validation year in the training rows"
    ens = fit_ensemble(rows(train_years), cfg, seeds)
    return predict(ens, rows(val_years)), float(np.mean(ens["epochs"]))


def auc(years, scores):
    y = rows(years)["HAB"]
    return roc_auc_score(y, scores) if y.nunique() == 2 else np.nan


def logreg(train_years, val_years):
    return utils.score(utils.fit_logreg(rows(train_years), SPEC, 5.0), rows(val_years))


# ---------------------------------------------------------------- verification

def verify(pool):
    cfg = {"hidden": (16,), "alpha": 1.0, "dropout": 0.0}
    train, val = rows(YEARS[:-1]), rows([2019])
    a, b = fit_ensemble(train, cfg, [3]), fit_ensemble(train, cfg, [3])
    same_seed = np.array_equal(predict(a, val), predict(b, val))
    one_by_one = np.concatenate([predict(a, val.iloc[[i]]) for i in range(len(val))])
    batch_invariant = np.allclose(one_by_one, predict(a, val), atol=1e-6)
    print(f"same seed, identical predictions: {same_seed} | one row at a time equals the batch: {batch_invariant}")
    assert same_seed and batch_invariant
    torch_auc = [auc(b, s) for b, (s, _) in zip(BLOCKS, pool.map(job, [([y for y in YEARS if y not in b], b, cfg, range(10)) for b in BLOCKS]))]
    sk_auc = []
    for b in BLOCKS:
        tr, va = rows([y for y in YEARS if y not in b]), rows(b)
        st = utils.fit_state(tr, SPEC)
        p = np.mean([MLPClassifier((16,), alpha=1.0, max_iter=2000, random_state=s).fit(utils.transform(tr, st), tr["HAB"])
                     .predict_proba(utils.transform(va, st))[:, 1] for s in range(10)], axis=0)
        sk_auc.append(auc(b, p))
    print("3-year blocks, (16,) alpha=1, 10 seeds:  PyTorch", np.round(torch_auc, 3).tolist(), f"mean {np.mean(torch_auc):.3f}",
          "| sklearn", np.round(sk_auc, 3).tolist(), f"mean {np.mean(sk_auc):.3f}")
    return {"same_seed": same_seed, "batch_invariant": batch_invariant, "torch_blocks": torch_auc, "sklearn_blocks": sk_auc}


# ---------------------------------------------------------------- nested evaluation

def main():
    global COMP
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-inner", type=int, default=10, help="networks per ensemble in the inner folds")
    parser.add_argument("--k-outer", type=int, default=30, help="networks per ensemble in the outer refits")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    grid = GRID[:3] if args.quick else GRID
    if args.quick:
        args.k_inner, args.k_outer = 2, 3
    COMP = load()
    start = time.time()
    evals = ([("loyo", [y for y in YEARS if y != v], (v,)) for v in YEARS] + [("block", [y for y in YEARS if y not in b], b) for b in BLOCKS]
             + [("2020", YEARS, (2020,))])
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worker) as pool:
        checks = verify(pool)
        inner_keys, inner_jobs = [], []
        for e, (kind, train_years, val_years) in enumerate(evals):
            for t in train_years:
                inner_train = [y for y in train_years if y != t]
                if rows([t])["HAB"].nunique() == 2 and rows(inner_train)["HAB"].nunique() == 2:
                    for c, cfg in enumerate(grid):
                        inner_keys.append((e, c, t)); inner_jobs.append((inner_train, (t,), cfg, range(args.k_inner)))
        inner = [auc((k[2],), s) for k, (s, _) in zip(inner_keys, pool.map(job, inner_jobs, chunksize=4))]
        table = pd.DataFrame(inner_keys, columns=["eval", "cfg", "year"]).assign(auc=inner).groupby(["eval", "cfg"])["auc"].mean()
        picks = {e: int(table.loc[e].idxmax()) for e in range(len(evals))}
        outer = list(pool.map(job, [(tr, va, grid[picks[e]], range(args.k_outer)) for e, (_, tr, va) in enumerate(evals)]))
    print(f"\nnested selection and refits: {(time.time() - start) / 60:.1f} min")

    res = []
    for e, ((kind, tr, va), (scores, epochs)) in enumerate(zip(evals, outer)):
        res.append({"kind": kind, "val": va, "nn": auc(va, scores), "lr": auc(va, logreg(tr, va)), "cfg": grid[picks[e]],
                    "inner": float(table.loc[e].max()), "epochs": epochs, "scores": scores})
    loyo = pd.DataFrame([r for r in res if r["kind"] == "loyo" and not np.isnan(r["nn"])]).set_index(pd.Index([r["val"][0] for r in res if r["kind"] == "loyo" and not np.isnan(r["nn"])]))
    lo = [r for r in res if r["kind"] == "loyo"]
    done = rows(YEARS)
    nn_oof = pd.Series(np.concatenate([r["scores"] for r in lo]), index=np.concatenate([rows(r["val"]).index for r in lo]))
    lr_oof = pd.concat([pd.Series(logreg([y for y in YEARS if y != r["val"][0]], r["val"]), index=rows(r["val"]).index) for r in lo])
    gain = loyo["nn"] - loyo["lr"]
    print("\nLEAVE-ONE-YEAR-OUT (nested)   year: LR C=5 | PyTorch MLP | chosen configuration")
    for y, r in loyo.iterrows():
        print(f"  {y}: {r['lr']:.3f} | {r['nn']:.3f} ({r['nn'] - r['lr']:+.3f}) | {r['cfg']}, epochs {r['epochs']:.0f}")
    print(f"  mean per year: {loyo['lr'].mean():.3f} | {loyo['nn'].mean():.3f} ({gain.mean():+.3f}), years worse: {(gain < -0.005).sum()}")
    print(f"  pooled: {roc_auc_score(done['HAB'], lr_oof.loc[done.index]):.3f} | {roc_auc_score(done['HAB'], nn_oof.loc[done.index]):.3f}")
    print("  criterion (+0.02 per year, at most 1 year worse):", bool(gain.mean() >= 0.02 and (gain < -0.005).sum() <= 1))
    bl = [r for r in res if r["kind"] == "block"]
    print("\n3-YEAR BLOCKS (nested, one pooled AUC per block, like the real test)")
    for r in bl:
        print(f"  {r['val'][0]}-{r['val'][-1]}: LR {r['lr']:.3f} | MLP {r['nn']:.3f} ({r['nn'] - r['lr']:+.3f}) | {r['cfg']}")
    d = np.array([r["nn"] - r["lr"] for r in bl])
    print(f"  mean: LR {np.mean([r['lr'] for r in bl]):.3f} | MLP {np.mean([r['nn'] for r in bl]):.3f} ({d.mean():+.3f}), MLP better in {(d > 0.005).sum()} of {len(d)}, worse in {(d < -0.005).sum()}")
    r20 = res[-1]
    print(f"\n2020, context only (configuration chosen on 2013-2019: {r20['cfg']}): LR {r20['lr']:.3f} | MLP {r20['nn']:.3f}")

    if not args.quick:                                          # final ensemble on 2013-2020, kept for reference only
        final =fit_ensemble(rows(YEARS + [2020]), r20["cfg"], range(args.k_outer))
        torch.save({"cfg": final["cfg"], "state": final["state"], "nets": [m.state_dict() for m in final["nets"]]}, OUT_MODEL)
        print("final ensemble (2013-2020,", len(final["nets"]), "networks) saved to", OUT_MODEL)
    for r in res:
        r.pop("scores")
    json.dump({"verification": checks, "results": res, "grid": grid, "k_inner": args.k_inner, "k_outer": args.k_outer},
              open(OUT_JSON, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved", OUT_JSON)


if __name__ == "__main__":
    main()
