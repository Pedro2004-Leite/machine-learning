"""Task 1: probability of a HAB event for the day received."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(model, day_and_lag_features_df):
    """Probability of class 1 (HAB), as an array of shape (1,)."""
    if model["family"] == "Extra Trees":
        return utils.score_extra_trees(model, day_and_lag_features_df)
    return utils.score_calibrated(model, day_and_lag_features_df)
