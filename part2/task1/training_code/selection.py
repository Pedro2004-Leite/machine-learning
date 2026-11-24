"""Model selection by nested leave-one-year-out cross-validation (training only, not needed by predict).

The candidates are a small grid of Logistic Regressions (C) and Extra Trees (min_samples_leaf, max_features,
class_weight). Choosing among them is part of the procedure that is evaluated:

    for each outer year (left out):
        training = the other years
        inner folds: leave-one-year-out inside the training years  ->  pick the best candidate
        refit the pick on the training years, and score the outer year (it never influenced the pick)

select() is the inner step, nested() is the whole procedure, and the final model is chosen by running select()
on all the training years.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import cv
import utils

C_GRID = [0.01, 0.1, 1, 10, 100]
ET_GRID = [{"min_samples_leaf": m, "max_features": f, "class_weight": w}
           for m in (1, 3, 5, 10) for f in ("sqrt", 0.5) for w in (None, "balanced")]


def candidates(spec=utils.DEFAULT_SPEC, n_estimators=1000, seed=0):
    """name -> {"family", "params", "fit_score(train, val)"}."""
    cands = {}
    for C in C_GRID:
        cands[f"Logistic Regression, C={C:g}"] = {
            "family": "Logistic Regression", "params": {"C": C},
            "fit_score": lambda tr, va, C=C: utils.score(utils.fit_logreg(tr, spec, C), va)}
    for p in ET_GRID:
        params = {**p, "n_estimators": n_estimators, "cluster": "number"}
        name = f"Extra Trees, leaf={p['min_samples_leaf']}, features={p['max_features']}, weights={p['class_weight'] or 'none'}"
        cands[name] = {
            "family": "Extra Trees", "params": params,
            "fit_score": lambda tr, va, params=params: utils.score_extra_trees(utils.fit_extra_trees(tr, **params, seed=seed), va)}
    return cands


def select(train, cands):
    """Inner step: mean leave-one-year-out AUC of every candidate inside the training rows."""
    rows = []
    for name, cand in cands.items():
        auc, _ = cv.loyo(train, cand["fit_score"])
        rows.append({"candidate": name, "family": cand["family"], "inner_auc": auc.mean()})
    return pd.DataFrame(rows).set_index("candidate")


def best_of(table):
    """The best candidate of each family and the best overall (the first one wins a tie)."""
    by_family = table.groupby("family")["inner_auc"].idxmax().to_dict()
    return by_family, table["inner_auc"].idxmax()


def nested(df, cands, outer_years=None):
    """Whole procedure. Returns the outer AUC of each year, the pooled AUC and what each outer fold picked.

    Columns: one per family (the best candidate of that family) and "choice" (the best overall).
    """
    outer_years = sorted(df["Year"].unique()) if outer_years is None else outer_years
    families = sorted({c["family"] for c in cands.values()})
    keys = families + ["choice"]
    auc = pd.DataFrame(np.nan, index=outer_years, columns=keys)
    oof = {k: pd.Series(np.nan, index=df.index) for k in keys}
    picks = []
    for year in outer_years:
        train, val = df[df["Year"] != year], df[df["Year"] == year]
        by_family, best = best_of(select(train, cands))
        scores = {name: cands[name]["fit_score"](train, val) for name in set(by_family.values()) | {best}}
        for family in families:
            oof[family].loc[val.index] = scores[by_family[family]]
        oof["choice"].loc[val.index] = scores[best]
        picks.append({"year": year, **by_family, "choice": best})
        if val["HAB"].nunique() == 2:
            for key in keys:
                auc.loc[year, key] = roc_auc_score(val["HAB"], oof[key].loc[val.index])
    pooled = pd.Series({k: roc_auc_score(df["HAB"], oof[k]) for k in keys})
    return auc, pooled, pd.DataFrame(picks).set_index("year")
