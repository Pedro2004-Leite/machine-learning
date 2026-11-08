import pickle
from pathlib import Path
import pandas as pd
import numpy as np

def load_model(model_path):
    # Fixes the Permission Denied error by ensuring it targets the file inside the directory
    actual_path = Path(model_path)
    if actual_path.is_dir():
        actual_path = actual_path / "model.pkl"
        
    with open(actual_path, "rb") as f:
        return pickle.load(f)

def predict(model, day_and_lag_features_df):
    df = day_and_lag_features_df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    
    features_list = model["features"]
    season_map = model["season_map"]
    chl_median = model["chl_median"]
    chl_unc_median = model["chl_unc_median"]
    X_mean = model["X_mean"]
    X_std = model["X_std"]
    
    W1, b1 = model["W1"], model["b1"]
    W2, b2 = model["W2"], model["b2"]
    W3, b3 = model["W3"], model["b3"]
    
    if 'Season' in df.columns:
        df['Season_num'] = df['Season'].map(season_map).fillna(0).astype(int)
        
    if 'CHL' in df.columns:
        df['CHL'] = df['CHL'].fillna(chl_median)
    if 'CHL_uncertainty' in df.columns:
        df['CHL_uncertainty'] = df['CHL_uncertainty'].fillna(chl_unc_median)
        
    doy_col = 'YDOY' if 'YDOY' in df.columns else 'DOY'
    if doy_col not in df.columns and 'DOY' in features_list:
        df['DOY'] = df['DOY'] if 'DOY' in df.columns else 0.0
        
    X_raw = df[features_list].to_numpy().astype(np.float32)
    X_scaled = (X_raw - X_mean) / X_std
    
    layer1 = X_scaled @ W1 + b1
    layer1 = np.maximum(0, layer1)
    
    layer2 = layer1 @ W2 + b2
    layer2 = np.maximum(0, layer2)
    
    logits = layer2 @ W3 + b3
    probs = 1 / (1 + np.exp(-logits.flatten()))
    
    preds = (probs >= 0.5).astype(np.int64)
    return preds