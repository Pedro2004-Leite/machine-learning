"""Cross-validation helpers used by the notebooks and by train.py (training only, not needed by predict).

Leave-one-year-out: each fold leaves one whole calendar year out, because rows of the same date
(and of the same year) look alike and must never be split between training and validation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def loyo(df, fit_score, years=None):
    """fit_score(train, val) -> scores for the rows of val.

    Returns the AUC of each year (NaN when the year has a single class, e.g. 2014 and 2017)
    and the out-of-fold score of every row, so the pooled AUC can be computed as well.
    """
    years = sorted(df["Year"].unique()) if years is None else years
    auc, oof = {}, pd.Series(np.nan, index=df.index)
    for year in years:
        train, val = df[df["Year"] != year], df[df["Year"] == year]
        oof.loc[val.index] = fit_score(train, val)
        auc[year] = roc_auc_score(val["HAB"], oof.loc[val.index]) if val["HAB"].nunique() == 2 else np.nan
    return pd.Series(auc, name="AUC"), oof


def walk_forward(df, fit_score, years=None):
    """Expanding window: each year is predicted with a model trained only on the earlier years.

    A harder check than leave-one-year-out (the model never sees the future), with little training
    data in the first folds. Returns the AUC of each year and the out-of-fold scores (NaN for the first year).
    """
    all_years = sorted(df["Year"].unique())
    years = all_years[1:] if years is None else years
    auc, oof = {}, pd.Series(np.nan, index=df.index)
    for year in years:
        train, val = df[df["Year"] < year], df[df["Year"] == year]
        oof.loc[val.index] = fit_score(train, val)
        auc[year] = roc_auc_score(val["HAB"], oof.loc[val.index]) if val["HAB"].nunique() == 2 else np.nan
    return pd.Series(auc, name="AUC"), oof


def pooled_auc(df, oof):
    """AUC of all out-of-fold scores together (the professor probably scores 2021-2023 in one go)."""
    return roc_auc_score(df["HAB"], oof)
