"""Prediction-time helpers shared by my_model.py and training_code/train.py.
All values are obrigatory 
Model state: 
    "estimator"      fitted predictive estimator
    "lag_spec"       {column: [lags]}
    "n_harmonics"    number of annual harmonics (1 if the key is absent)
    "feature_names"  ordered list, equal to feature_names(lag_spec, n_harmonics)
    "mean", "std" 
    "smearing"
    "fallback"
    "log_target"
    "clip_limits"   
"""

import numpy as np
import pandas as pd


# Names of the columns of the dataframe and of the lag_spec 
DATE = "date"
TARGET = "chlorophyll_a_mg_m3"
EXOGENOUS = [
    "sst_c",
    "par_umol_m2_s",
    "nitrate_umol_l",
    "wind_speed_m_s",
    "upwelling_index",
    "mixed_layer_depth_m",
    "salinity_psu",
    "current_speed_m_s",
    "river_discharge_index",
    "cloud_fraction",
    "surface_pressure_hpa",
    "turbidity_ntu",
]
YEAR_PERIOD = 365.25 

def check_lag_spec(lag_spec):
    """
    This funcion raises an exception if the lag_spec is not in accordance 
    """
    # Check if lag_spec in an empty dict
    if not lag_spec:
        raise ValueError("Empty lag config")
    
    for key, values in lag_spec.items(): 
        # Test if the configuration is empty 
        if not values: 
            raise ValueError(f"Empty lag config for variable {key}")
        
        # Test if there are repeated lags 
        if (len(set(values)) != len(values)): 
            raise ValueError("Repeated lag values")
        
        #Check for unknown columns 
        if key not in (TARGET, *EXOGENOUS): 
            raise ValueError("Unknown column name")
        
    
        for lag in values:
            if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool):  
                raise ValueError("Lag should be type int")
            if lag < 0: 
                raise ValueError("Value of lag less that 0")
            if key == TARGET and lag == 0: 
                raise ValueError("Value of the lag associated to the target cannot the 0, data leakage") 


def sorted_frame(history_df, day_features_df=None):
    """
    This function returns a sorted dataframe with the added target column.
    all sorted by day. 
    """
    #Deals with not receiving the day_features dataframe
    parts = [history_df]
    if day_features_df is not None: 
        parts.append(day_features_df)

    merged = pd.concat(parts, ignore_index=True, sort = False) #Does not modify inplace. 
    merged[DATE] = pd.to_datetime(merged[DATE])  
    merged = merged.sort_values(DATE, kind = "stable").reset_index(drop=True) 

    # If the Target column does not exists, we created with nan values 
    
    if TARGET not in merged.columns: 
        merged[TARGET] = np.nan 
        
    return merged 


def feature_names(lag_spec, n_harmonics=1):
    """
    Creates list of feature names 
    """
    # Name of the lag in accordance with the professor definition 
    names = [f"{name}__lag_{lag}" for name, lags in lag_spec.items() for lag in lags]
    for n in range(1, n_harmonics + 1): 
        names.append(f"annual_sin_{n}")
        names.append(f"annual_cos_{n}") 
    
    return names


def build_features(frame, lag_spec, n_harmonics=1, log_target = True):
    """
    Receives already sorted frame, checks the lag_spec 
    """
    
    check_lag_spec(lag_spec)
    #Create a dict where the keys are the names of te lags 
    series = {name : [] for name in feature_names(lag_spec, n_harmonics)} 
    
    for col, lags in lag_spec.items(): 
        serie = pd.to_numeric(frame[col], errors="coerce")
        if col == TARGET and log_target:
            serie = serie.where(serie > 0) 
            serie = np.log(serie) 
        for lag in lags: 
            series[f"{col}__lag_{lag}"] = serie.shift(lag) 
            
    d = frame[DATE].dt.dayofyear # Obtain day of the year
    for n in range(1, n_harmonics + 1): 
        series[f"annual_sin_{n}"] = np.sin(2 * np.pi * n * d / YEAR_PERIOD)
        series[f"annual_cos_{n}"] = np.cos( 2 * np.pi * n * d / YEAR_PERIOD) 
    
    data = pd.DataFrame(series, index = frame.index) 
    
    return data 


def build_training_matrix(raw_df, lag_spec, n_harmonics=1, log_target = True):
    """
    Receives the raw frame, sorts it, builds the features, then drops the rows with Nan values 
    and gives back X, y and dates in separate Dataframe objects 
    """
    
    sorted_df = sorted_frame(raw_df)
    data = build_features(sorted_df, lag_spec, n_harmonics, log_target) 
    mask = data.notna().all(axis = 1) & (sorted_df[TARGET] > 0)
 
    X = data.loc[mask].reset_index(drop = True)
    y = sorted_df.loc[mask][TARGET].reset_index(drop = True)
    dates = sorted_df.loc[mask][DATE].reset_index(drop = True)

    return X, y, dates     
    
    

def build_lagged_row(history_df, day_features_df, lag_spec, n_harmonics=1, log_target = True):
    """
    Computes the first day of day_features. Gets all the non relevant days from history, 
    sorts the two, creating a comun dataframe and then we build the features, taking the last line 
    of the dataframe.
    """
    
    if len(day_features_df) != 1: 
        raise ValueError("day features should be of dimension 1")
    timestamp = pd.to_datetime(day_features_df[DATE]).iloc[0]
    mask = pd.to_datetime(history_df[DATE]) < timestamp # Obtain passed dates 
    passed = history_df[mask] 
    sorted = sorted_frame(passed, day_features_df) 
    data = build_features(sorted, lag_spec, n_harmonics, log_target) 
    
    return data.tail(1).reset_index(drop = True)  


def winsorize(values, lo, hi):
    """Clip values to limits. Limits are learned on training data and stored."""
    return np.clip(values, a_min=lo, a_max=hi) 

def standardize(x, mean, std):
    """Apply standardization"""
    return (x - mean)/std


def target_forward(y, log_target = True):
    """Target transform"""
    if log_target: 
        return np.log(y)
    else: 
        return y

def target_inverse(z, log_target = True, smearing = 1.0):
    """Target inverse transformation with smearing"""
    
    if log_target: 
        return smearing * np.exp(z) 
    else: 
        return z


def apply_preprocessing(X, state):
    """Prepares a feature table for the estimator, using the parameters in the 
    state of the estimator, that are learned during training 
    """
    names = list(state["feature_names"])
    values = X[names].to_numpy(dtype=float, copy=True)
    limits = state["clip_limits"] 
    if limits is not None:
        lo, hi = limits
        lag_columns = [i for i, name in enumerate(names) if name.startswith(f"{TARGET}__lag_")]
        if lag_columns:
            for col in lag_columns: 
                values[:, col] = winsorize(values[:, col], lo, hi)
    scaled = standardize(values, state["mean"], state["std"])
    return np.where(np.isfinite(scaled), scaled, 0.0)


def predict_from_features(model, X):
    """
    This function predicts the next day clorofile, based on the exogenous of the day and 
    before ouputs and exogenous variables; 
    It reaceives X from build_lagged_row or from normal training
    """
    X = apply_preprocessing(X, model)
    y = model["estimator"].predict(X) 
    y = target_inverse(y, log_target=model["log_target"], smearing = model["smearing"])
    y = np.where(np.isfinite(y), y, model["fallback"])
    y = np.maximum(y, 0.0)
    
    return y

