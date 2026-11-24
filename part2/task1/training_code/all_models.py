"""Every model trained in the project, scored the same way (used by notebook 05, training only, not needed by predict).

compare() returns the ROC AUC of each model in each leave-one-year-out fold, the pooled AUC, the 2020 AUC
(training 2013-2019) and the out-of-fold scores. Except for the Logistic Regression and the Extra Trees (chosen in
notebook 04), the hyperparameters are the ones fixed in the exploration, not tuned in a nested way.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import cv
import utils

SPEC = utils.DEFAULT_SPEC


def logreg(spec=SPEC, C=5.0):
    return lambda tr, va: utils.score(utils.fit_logreg(tr, spec, C), va)


def design(tr, va):
    state = utils.fit_state(tr, SPEC)
    return utils.transform(tr, state), utils.transform(va, state)


def extra_trees(tr, va):
    return utils.score_extra_trees(utils.fit_extra_trees(tr, n_estimators=1000, min_samples_leaf=5, max_features="sqrt"), va)


def svm(tr, va):
    A, B = design(tr, va)
    return SVC(C=1.0, gamma="scale", class_weight="balanced").fit(A, tr["HAB"]).decision_function(B)


def mlp(tr, va):
    A, B = design(tr, va)
    return np.mean([MLPClassifier((4,), alpha=0.01, max_iter=2000, random_state=s).fit(A, tr["HAB"]).predict_proba(B)[:, 1]
                    for s in range(3)], axis=0)


def knn(tr, va):
    A, B = design(tr, va)
    return KNeighborsClassifier(25).fit(A, tr["HAB"]).predict_proba(B)[:, 1]


def tree_X(d):
    x = pd.DataFrame({"SST": d["SST"], "UI": d["UI"]})
    x["cos1"], x["sin1"] = utils.seasonal_terms(d["Date"], ["cos1", "sin1"])
    x["logCHL"], x["unc"], x["cluster"] = np.log(d["CHL"]), d["CHL_uncertainty"], d["spatial_cluster"]
    return x


def random_forest(tr, va):
    forest = RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample",
                                    random_state=0, n_jobs=-1)
    return forest.fit(tree_X(tr), tr["HAB"]).predict_proba(tree_X(va))[:, 1]


def boosting(tr, va):
    model = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=80, min_samples_leaf=20,
                                           l2_regularization=1.0, class_weight="balanced", random_state=0)
    return model.fit(tree_X(tr), tr["HAB"]).predict_proba(tree_X(va))[:, 1]


def models():
    no_env = {**SPEC, "SST": False, "UI": False, "CHL": None, "unc": None}
    return {
        "só sazonalidade": logreg({**no_env, "cluster": None}),
        "só ambientais": logreg({**SPEC, "season": [], "cluster": None}),
        "sazonalidade + cluster": logreg(no_env),
        **{f"Logistic Regression, C={C:g}": logreg(SPEC, C) for C in (0.01, 0.1, 1, 5, 10, 100)},
        "Extra Trees": extra_trees, "Random Forest": random_forest, "Gradient Boosting": boosting,
        "SVM (RBF)": svm, "MLP (4 neurónios)": mlp, "kNN (k=25)": knn,
    }


def compare(comp, c20):
    """Table (one row per model: AUC of each year, "media", "agrupado", "2020") and the out-of-fold scores."""
    rows, oofs = {}, {}
    for name, fit_score in models().items():
        auc, oof = cv.loyo(comp, fit_score)
        rows[name] = {**{int(y): v for y, v in auc.dropna().items()}, "media": auc.mean(),
                      "agrupado": cv.pooled_auc(comp, oof), "2020": roc_auc_score(c20["HAB"], fit_score(comp, c20))}
        oofs[name] = oof
    return pd.DataFrame(rows).T, oofs


def confusion(y, p, threshold=0.5):
    """Confusion matrix [[TN, FP], [FN, TP]] with recall, specificity and precision."""
    cm = confusion_matrix(y, (np.asarray(p) >= threshold).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"cm": cm, "recall": tp / (tp + fn), "specificity": tn / (tn + fp), "precision": tp / (tp + fp)}
