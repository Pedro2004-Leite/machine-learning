"""Training of the final Task 1 model (HAB).

    python training_code/train.py classification_task1_train.xlsx

1. Read the data and sort it by date.
2. Keep the complete rows (CHL present), since the test set only has complete rows.
3. Optimise and compare two algorithms, Logistic Regression (C) and Extra Trees (leaf size, features per split,
   class weights), with a nested leave-one-year-out cross-validation on 2013-2019: for each year left out,
   the hyperparameters are chosen by a leave-one-year-out on the other years, and the chosen model is scored on the
   year left out. The algorithm with the best mean AUC on the years left out is kept.
4. Choose its hyperparameters with a leave-one-year-out on 2013-2019, fit it with every year (2013-2020) and
   save it to best_model/model.pkl.

The same functions (utils.py, cv.py, selection.py) are used in the notebooks, which also compare other algorithms
(Random Forest, Gradient Boosting, SVM, MLP, kNN). The full run takes a few minutes.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "best_model"))
import selection  # noqa: E402
import utils  # noqa: E402

C_GRID = [0.01, 0.1, 1, 5, 10, 100]


def read_data(path):
    df = pd.read_excel(path) if str(path).endswith(".xlsx") else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df[df["CHL"].notna()]


def candidates():
    """Logistic Regressions (one per C) and Extra Trees (one per hyperparameter combination)."""
    cands = {}
    for C in C_GRID:
        cands[f"Logistic Regression, C={C:g}"] = {
            "family": "Logistic Regression", "params": {"C": C},
            "fit_score": lambda tr, va, C=C: utils.score(utils.fit_logreg(tr, utils.DEFAULT_SPEC, C), va)}
    for name, cand in selection.candidates().items():
        if cand["family"] == "Extra Trees":
            cands[name] = cand
    return cands


def fit(cand, df):
    if cand["family"] == "Logistic Regression":
        model = utils.fit_logreg(df, utils.DEFAULT_SPEC, cand["params"]["C"])
    else:
        model = utils.fit_extra_trees(df, **cand["params"])
    model["family"] = cand["family"]
    return model


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else HERE.parents[1] / "data" / "classification_task1_train.xlsx"
    df = read_data(path)
    train = df[df["Year"] < 2020]
    cands = candidates()

    auc, pooled, picks = selection.nested(train, cands)
    families = ["Logistic Regression", "Extra Trees"]
    print("Nested leave-one-year-out, AUC of each year left out:")
    print(auc[families].dropna().round(3).to_string())
    print("mean per year:", auc[families].mean().round(3).to_dict())
    print("pooled:", pooled[families].round(3).to_dict())
    family = auc[families].mean().idxmax()

    table = selection.select(train, {n: c for n, c in cands.items() if c["family"] == family})
    best = table["inner_auc"].idxmax()
    print(f"\nchosen: {family}; leave-one-year-out on 2013-2019:")
    print(table["inner_auc"].round(3).to_string())

    model = fit(cands[best], df)
    with open(HERE.parent / "best_model" / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\nfinal model: {best}, trained on {len(df)} rows of 2013-2020 ({df['HAB'].sum()} HAB)")


if __name__ == "__main__":
    main()
