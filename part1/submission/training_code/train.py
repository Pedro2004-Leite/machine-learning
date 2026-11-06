"""
This code should redo all of the training for the final obtained model.
In order to take maximum advantage the refit is used for the interval 
of years from 2015 - 2023 (all of the training data). 
The configuration for the model to be used in the submission is the one 
obtained in the experiments notebook which is a Lasso estimator.  
"""
import sys
import pickle
from pathlib import Path
import pandas as pd

SUBMISSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION / "best_model"))

import utils
import walk_forward

DATA = SUBMISSION.parent / "data" / "chlorophyll_student_2015_2023.csv"

raw = pd.read_csv(DATA)
raw[utils.DATE] = pd.to_datetime(raw[utils.DATE])
raw = raw.sort_values(utils.DATE).reset_index(drop=True)

final_config = {
    "lag_spec": {
        utils.TARGET: [1],
        "nitrate_umol_l": [0, 1, 2, 3],
        "upwelling_index": [0, 1, 2, 3],
        "wind_speed_m_s": [0, 1],
    },
    "n_harmonics": 2,
    "log_target": True,
    "clip_percentiles": (0.5, 99.5),
    "smearing": False,
    "estimator": ("lasso", {"alpha": 1e-3}),
}


X, y, _ = utils.build_training_matrix(raw, final_config["lag_spec"], final_config["n_harmonics"], 
                                          final_config["log_target"])
#Train model in all data 
model_state = walk_forward.train(X, y, final_config)

#Confirm if everything is correct: 
print(model_state["estimator"])
print(model_state["clip_limits"]) 
print(model_state["mean"]) 
print(model_state["std"])
print(model_state["fallback"])

MODEL_PATH = SUBMISSION / "best_model" / "model.pkl"

with MODEL_PATH.open("wb") as handle:
    pickle.dump(model_state, handle)

print("Saved", MODEL_PATH) 


ridge_config = dict(final_config, estimator=("ridge", {"alpha": 1.0}))

X, y, _ = utils.build_training_matrix(raw, ridge_config["lag_spec"], ridge_config["n_harmonics"], 
                                          ridge_config["log_target"])
#Train model in all data 
model_state = walk_forward.train(X, y, ridge_config)

#Confirm if everything is correct: 
print(model_state["estimator"])
print(model_state["clip_limits"]) 
print(model_state["mean"]) 
print(model_state["std"])
print(model_state["fallback"])

MODEL_PATH = SUBMISSION / "best_model" / "model_ridge.pkl"

with MODEL_PATH.open("wb") as handle:
    pickle.dump(model_state, handle)

print("Saved", MODEL_PATH)