"""Feature construction and models for Task 1, shared by the notebooks, training_code/train.py and my_model.py.

The cross-validation folds, the final training and predict all build the data matrix with fit_state / transform.

Model state (plain dict, saved in model.pkl):
    "spec"       feature configuration (see DEFAULT_SPEC)
    "clusters"   spatial clusters seen in training, in order (columns of the one-hot block)
    "medians"    training medians of CHL and CHL_uncertainty (used only when a NaN arrives)
    "mean","std" z-score of the numeric block, fitted on the training rows
    "estimator"  fitted LogisticRegression
    "prior"      HAB rate of the training rows
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# YDOY in the dataset is cos(2*pi*day_of_year/365), also in leap years (day 366 wraps around).
DAYS = 365

DEFAULT_SPEC = {
    "season": ["cos1", "sin1", "cos2", "sin2"],   # terms built from Date: cos/sin of h * 2*pi*day/365
    "SST": True,
    "UI": True,
    "CHL": "log",          # "log", "raw" or None
    "unc": "raw",          # CHL_uncertainty: "log", "raw" or None (skewness 0.50, log changes nothing: notebook 02)
    "cluster": "onehot",   # "onehot", "numeric" or None
    "year": False,         # Year is never a feature in the real model (test years are new)
    "chl_flag": False,     # extra 0/1 column: CHL was missing (only used in the imputation comparison)
    "zscore": True,        # False only to show what the z-score changes (ablation)
}


def seasonal_terms(dates, terms):
    """cos/sin terms of the day of the year, e.g. ["cos1", "sin1", "cos2", "sin2"]."""
    day = pd.to_datetime(dates).dt.dayofyear.to_numpy()
    angle = 2 * np.pi * day / DAYS
    out = []
    for term in terms:
        h = int(term[3:])
        out.append(np.cos(h * angle) if term[:3] == "cos" else np.sin(h * angle))
    return out


def fallback_medians(train):
    """Medians used only if a row arrives with CHL or CHL_uncertainty missing (never in the real test)."""
    return {"CHL": float(train["CHL"].median()), "CHL_uncertainty": float(train["CHL_uncertainty"].median())}


def numeric_block(df, spec, medians):
    """Numeric columns before the z-score, in a fixed order. Missing CHL/uncertainty take the training median."""
    cols = seasonal_terms(df["Date"], spec["season"]) if spec["season"] else []
    names = list(spec["season"])
    if spec["SST"]:
        cols.append(df["SST"].to_numpy(float)); names.append("SST")
    if spec["UI"]:
        cols.append(df["UI"].to_numpy(float)); names.append("UI")
    for key, column in (("CHL", "CHL"), ("unc", "CHL_uncertainty")):
        if spec[key] is not None:
            x = df[column].fillna(medians[column]).to_numpy(float)
            cols.append(np.log(x) if spec[key] == "log" else x)
            names.append(f"{spec[key]}_{column}")
    if spec["chl_flag"]:
        cols.append(df["CHL"].isna().to_numpy(float)); names.append("CHL_missing")
    if spec["cluster"] == "numeric":
        cols.append(df["spatial_cluster"].to_numpy(float)); names.append("cluster_number")
    if spec["year"]:
        cols.append(df["Year"].to_numpy(float)); names.append("Year")
    return np.column_stack(cols), names


def fit_state(train, spec):
    """Everything learned from the training rows: clusters seen, medians, z-score mean and std."""
    medians = fallback_medians(train)
    block, names = numeric_block(train, spec, medians)
    mean, std = block.mean(axis=0), block.std(axis=0)
    if not spec.get("zscore", True):
        mean, std = np.zeros(block.shape[1]), np.ones(block.shape[1])
    return {"spec": spec, "clusters": sorted(int(c) for c in train["spatial_cluster"].unique()),
            "medians": medians, "mean": mean, "std": std, "names": names}


def transform(df, state):
    """Design matrix: z-scored numeric block, then the one-hot clusters (a new cluster gives all zeros)."""
    spec = state["spec"]
    block, _ = numeric_block(df, spec, state["medians"])
    parts = [(block - state["mean"]) / state["std"]] if block.shape[1] else []
    if spec["cluster"] == "onehot":
        parts.append(np.column_stack([(df["spatial_cluster"].to_numpy() == c).astype(float) for c in state["clusters"]]))
    return np.hstack(parts)


def feature_names(state):
    names = list(state["names"])
    if state["spec"]["cluster"] == "onehot":
        names += [f"cluster_{c}" for c in state["clusters"]]
    return names


def fit_logreg(train, spec=DEFAULT_SPEC, C=3.0, class_weight="balanced"):
    """Logistic regression (L2, balanced class weights by default), fitted on the training rows only."""
    state = fit_state(train, spec)
    estimator = LogisticRegression(solver="liblinear", penalty="l2", C=C, class_weight=class_weight)
    estimator.fit(transform(train, state), train["HAB"].to_numpy())
    state["estimator"] = estimator
    state["prior"] = float(train["HAB"].mean())          # HAB rate of the training rows
    return state


def score(model, df):
    """Probability of class 1 (HAB) for each row of df (inflated when class_weight is balanced)."""
    return model["estimator"].predict_proba(transform(df, model))[:, 1]


def score_calibrated(model, df):
    """Same ranking as score(), with the intercept moved back to the HAB rate of the training rows.

    class_weight="balanced" trains as if HAB were 50% of the rows, so the raw probabilities are too high.
    Adding log(prior / (1 - prior)) to the log-odds undoes that (King and Zeng's prior correction).
    The AUC does not change, and the probabilities become realistic.
    """
    log_odds = model["estimator"].decision_function(transform(df, model))
    return 1.0 / (1.0 + np.exp(-(log_odds + np.log(model["prior"] / (1.0 - model["prior"])))))


# ---- Extra Trees: raw values, no z-score (a split only compares values of one feature) ----

def tree_matrix(df, medians, clusters=None):
    """SST, UI, cos/sin of the day, CHL and CHL_uncertainty (training medians fill NaN), and the cluster.

    clusters=None puts the cluster in as a number (column 5); a list of clusters gives one-hot columns at the end.
    """
    cos1, sin1 = seasonal_terms(df["Date"], ["cos1", "sin1"])
    cluster = df["spatial_cluster"].to_numpy()
    chl, unc = df["CHL"].fillna(medians["CHL"]), df["CHL_uncertainty"].fillna(medians["CHL_uncertainty"])
    if clusters is None:
        cols = [df["SST"], df["UI"], cos1, sin1, cluster, chl, unc]
    else:
        cols = [df["SST"], df["UI"], cos1, sin1, chl, unc] + [(cluster == c).astype(float) for c in clusters]
    return np.column_stack(cols).astype(float)


def fit_extra_trees(train, n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight=None,
                    cluster="number", seed=0):
    medians = fallback_medians(train)
    clusters = sorted(int(c) for c in train["spatial_cluster"].unique()) if cluster == "onehot" else None
    estimator = ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                     max_features=max_features, class_weight=class_weight, random_state=seed)
    estimator.fit(tree_matrix(train, medians, clusters), train["HAB"].to_numpy())
    return {"medians": medians, "clusters": clusters, "estimator": estimator}


def score_extra_trees(model, df):
    """Probability of class 1 (HAB) from the Extra Trees."""
    return model["estimator"].predict_proba(tree_matrix(df, model["medians"], model["clusters"]))[:, 1]
