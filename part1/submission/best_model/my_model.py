"""Grading interface: load_model and predict. """


from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import utils

def load_model(model_path):
   
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)

def predict(model, history_df, day_features_df):
   X = utils.build_lagged_row(history_df, day_features_df, 
                              model["lag_spec"], model["n_harmonics"], model["log_target"])
   y_pred = utils.predict_from_features(model, X)
   
   return y_pred 


