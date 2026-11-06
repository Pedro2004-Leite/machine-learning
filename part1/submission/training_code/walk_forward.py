import numpy as np 
import sys 
from sklearn.linear_model import Ridge, Lasso
from pathlib import Path

#Make the import of utils possible 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "best_model"))

import utils 

def split(dates, test_years):
    idx_list = []
    for year in test_years: 
        train_idx = np.flatnonzero(dates.dt.year < year) 
        test_idx = np.flatnonzero(dates.dt.year == year)
        
        pair = (train_idx, test_idx)
        idx_list.append(pair)
        
    return idx_list 

def train(X, y, config): 
    
    y_model = utils.target_forward(y,config["log_target"])
    
    if config["clip_percentiles"] is not None: 
        lo, hi = np.percentile(y_model, config["clip_percentiles"])
    
    state = {"lag_spec": config["lag_spec"], 
              "n_harmonics":config["n_harmonics"],
              "log_target": config["log_target"], 
              "feature_names": utils.feature_names(config["lag_spec"], config["n_harmonics"]),
              "clip_limits": (lo, hi) if config["clip_percentiles"] is not None else None, 
              "mean": np.zeros(len(X.columns)), 
              "std": np.ones(len(X.columns)), 
              "smearing": 1.0,
              "fallback": np.median(y), 
              "estimator": None}
    
    X_buffer = utils.apply_preprocessing(X, state) 
    state["mean"] = np.mean(X_buffer, axis = 0)
    state["std"] = np.std(X_buffer, axis = 0)
    X_pre = utils.apply_preprocessing(X, state) 
    
    #Create the estimator with the specific hyperparameters in config  
    
    model_name, params = config["estimator"] 
    
    if model_name == "lasso": 
        model = Lasso(**params) 
    elif model_name == "ridge": 
        model = Ridge(**params)
    else: 
        raise RuntimeError("No model was provided in the config")
    
    model.fit(X_pre, y_model) 
    
    #Update state with the trained model 
    state["estimator"] = model
    
    if config["smearing"] and config["log_target"]: 
        y_pred = model.predict(X_pre)
        error = y_model - y_pred 
        state["smearing"] = float(np.mean(np.exp(error)))
        
    return state 
    
# Scores used: 

def rmse_score(y, y_pred): 
    return np.sqrt(np.sum((y - y_pred)**2)/len(y))

def r2_score(y, y_pred): 
    return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

def mae_score(y, y_pred): 
    return np.sum(np.abs(y - y_pred)) / len(y)
 
def measure(X, y, state): 
    """
    Makes the prediction with the trained model in the fold and computes the metric,
    used the predict function from utils 
    """
    y_pred = utils.predict_from_features(state, X) 
    rmse = rmse_score(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mae_score(y, y_pred)

    metrics = {"rmse":rmse, "r2": r2, "mae":mae}
    
    return y_pred, metrics  

def baselines(y_train, y_test):
    """
    This function is used to compute the baselines: 
    1. average    
    2. ar(1) with no learnable parameter    
    """ 
    y_1 = np.mean(y_train) 
    y_2 = y_test.shift(1)
    y_2.iloc[0] = y_train.iloc[-1] 
     
    metrics1 = {"rmse":rmse_score(y_test, y_1), "r2": r2_score(y_test, y_1), "mae":mae_score(y_test, y_1)}
    metrics2 = {"rmse":rmse_score(y_test, y_2), "r2": r2_score(y_test, y_2), "mae":mae_score(y_test, y_2)}
    
    baseline_metrics = {"train_mean": metrics1, "persistence": metrics2}
    
    return  baseline_metrics 
    
def repeat(raw_df, config, test_years): 
    
    results = dict() 
    r2_scores = list()
    
    X, y, dates = utils.build_training_matrix(raw_df, lag_spec = config["lag_spec"], n_harmonics=config["n_harmonics"], log_target = config["log_target"]) 
    idxs = split(dates, test_years=test_years)
    
    for (train_idx, test_idx), test_year in zip(idxs, test_years): 
        X_train = X.iloc[train_idx] 
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        state = train(X_train, y_train, config)
        y_pred, metrics = measure(X_test, y_test, state) 
        
        r2 = metrics["r2"]
        r2_scores.append(r2)
        
        baseline_metrics = baselines(y_train, y_test)
        
        
        results[f"{test_year}"] = {"metrics": metrics, "baselines": baseline_metrics, "y_pred": y_pred} 
        
    return np.mean(r2_scores), results 
    

        
        
        
    

    

    
    
