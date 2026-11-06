"""Task 1 interface. PLACEHOLDER dummy model that always predicts class 0.

Replace with the real predictor. Required public entry points: load_model, predict.
"""

import pickle

import numpy as np


def load_model(model_path):
    """Return the fitted predictor and plain numerical state."""
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def predict(model, day_and_lag_features_df):
    """Return a finite NumPy array with shape (1,)."""
    return np.array([model["constant_class"]])
