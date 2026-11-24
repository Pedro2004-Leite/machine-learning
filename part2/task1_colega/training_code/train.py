import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Enforce GPU acceleration if available, otherwise fallback to CPU
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
        
    df = pd.read_excel(data_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    
    # Chronological sorting alignment
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'spatial_cluster']).reset_index(drop=True)
    
    # 1. Missing Data Transformation (Imputation via median)
    chl_median = float(df['CHL'].median())
    chl_unc_median = float(df['CHL_uncertainty'].median())
    df['CHL'] = df['CHL'].fillna(chl_median)
    df['CHL_uncertainty'] = df['CHL_uncertainty'].fillna(chl_unc_median)
    
    # 2. Categorical Transformation (Ordinal mapping for Season)
    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df['Season_num'] = df['Season'].map(season_map).fillna(0).astype(int)
    
    doy_col = 'YDOY' if 'YDOY' in df.columns else 'DOY'
    features = ['Day', 'Month', 'Year', 'Season_num', doy_col, 'spatial_cluster', 'SST', 'UI', 'CHL', 'CHL_uncertainty']
    target_col = 'HAB'
    
    X_raw = df[features].to_numpy().astype(np.float32)
    y_raw = df[target_col].to_numpy().astype(np.float32).reshape(-1, 1)
    
    # 3. Continuous Data Transformation (Manual Z-score scaling parameters)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X_raw - X_mean) / X_std
    
    # Multilayer Perceptron Architecture with a linear output layer for BCEWithLogitsLoss
    class ClassificationNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1) # Sigmoid behavior handled inside the loss function
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
            
    model = ClassificationNet(len(features)).to(device)
    
    # Anomaly Detection Balancing Action (pos_weight takes care of the rare class)
    neg_counts = np.sum(y_raw == 0)
    pos_counts = np.sum(y_raw == 1)
    pos_weight_val = neg_counts / max(1.0, pos_counts)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.005) # AdamW works as an optimized stochastic gradient descent
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_raw, dtype=torch.float32).to(device)
    
    # Training Loop tracking history and anti-overfitting callbacks
    best_loss = float('inf')
    best_weights = None
    patience = 15
    patience_counter = 0
    history_loss = []
    
    max_iterations = 300
    print("Beginning training iterations...")
    
    for epoch in range(max_iterations):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        history_loss.append(current_loss)
        
        # Early Stopping Logic: do not stop on the first oscillation, look for steady degradation
        if current_loss < best_loss:
            best_loss = current_loss
            # Extract weights right before overfitting starts
            best_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Overfitting mitigation callback triggered. Early stopping at epoch {epoch}.")
            break
            
    # Load back the best parameters achieved right before the validation loss drifted
    final_state_dict = {k: torch.tensor(v).to(device) for k, v in best_weights.items()}
    model.load_state_dict(final_state_dict)
    model.eval()
    
    # Save the custom weights array dictionary into model.pkl (Sklearn Scalers/Pipelines are forbidden)
    state = {
        "W1": best_weights['fc1.weight'].T,
        "b1": best_weights['fc1.bias'],
        "W2": best_weights['fc2.weight'].T,
        "b2": best_weights['fc2.bias'],
        "W3": best_weights['fc3.weight'].T,
        "b3": best_weights['fc3.bias'],
        "features": features,
        "season_map": season_map,
        "chl_median": chl_median,
        "chl_unc_median": chl_unc_median,
        "X_mean": X_mean,
        "X_std": X_std,
        "history_loss": history_loss
    }
    
    output_dir = Path("best_model")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(state, f)
    print("Model serialization complete. Best plain numerical weights saved to best_model/model.pkl.")

if __name__ == "__main__":
    main()