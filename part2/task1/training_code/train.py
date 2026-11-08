import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Neural Network training allocated on device: {device}")
    
    # Robust multi-path lookup for the training spreadsheet
    possible_paths = [
        Path("classification_task1_train.xlsx"),
        Path("data/classification_task1_train.xlsx"),
        Path("../classification_task1_train.xlsx"),
        Path("../../classification_task1_train.xlsx"),
        Path("../data/classification_task1_train.xlsx")
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
            
    if data_path is None:
        raise FileNotFoundError("Could not find 'classification_task1_train.xlsx' in any standard location.")
        
    print(f"Training model using dataset from: {data_path.resolve()}")
    df = pd.read_excel(data_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'spatial_cluster']).reset_index(drop=True)
    
    # Explicit missing value processing via median computation
    chl_median = float(df['CHL'].median())
    chl_unc_median = float(df['CHL_uncertainty'].median())
    df['CHL'] = df['CHL'].fillna(chl_median)
    df['CHL_uncertainty'] = df['CHL_uncertainty'].fillna(chl_unc_median)
    
    # Explicit encoding for categorical season parameters
    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df['Season_num'] = df['Season'].map(season_map).fillna(0).astype(int)
    
    doy_col = 'YDOY' if 'YDOY' in df.columns else 'DOY'
    features = ['Day', 'Month', 'Year', 'Season_num', doy_col, 'spatial_cluster', 'SST', 'UI', 'CHL', 'CHL_uncertainty']
    target_col = 'HAB'
    
    X_raw = df[features].to_numpy().astype(np.float32)
    y_raw = df[target_col].to_numpy().astype(np.float32).reshape(-1, 1)
    
    # Explicit Z-score scaling derivation
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X_raw - X_mean) / X_std
    
    # PyTorch Classification Architecture
    class ClassificationNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
            
    model = ClassificationNet(len(features)).to(device)
    
    neg_counts = np.sum(y_raw == 0)
    pos_counts = np.sum(y_raw == 1)
    pos_weight_val = neg_counts / max(1.0, pos_counts)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_raw, dtype=torch.float32).to(device)
    
    # Run optimization sequence
    model.train()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
    # Weight matrices extraction block
    model.eval()
    state_dict = model.state_dict()
    
    state = {
        "W1": state_dict['fc1.weight'].cpu().numpy().T,
        "b1": state_dict['fc1.bias'].cpu().numpy(),
        "W2": state_dict['fc2.weight'].cpu().numpy().T,
        "b2": state_dict['fc2.bias'].cpu().numpy(),
        "W3": state_dict['fc3.weight'].cpu().numpy().T,
        "b3": state_dict['fc3.bias'].cpu().numpy(),
        "features": features,
        "season_map": season_map,
        "chl_median": chl_median,
        "chl_unc_median": chl_unc_median,
        "X_mean": X_mean,
        "X_std": X_std
    }
    
    output_dir = Path("best_model")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(state, f)
    print("Model serialization complete. Output directed to best_model/model.pkl.")

if __name__ == "__main__":
    main()