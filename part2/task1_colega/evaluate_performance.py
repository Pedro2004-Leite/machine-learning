import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# Ensure the best_model directory is in the path
sys.path.append(str(Path("best_model").resolve()))

try:
    import my_model
except ModuleNotFoundError:
    print("Error: Could not find 'best_model/my_model.py'. Verify project structure.")
    sys.exit(1)

def evaluate_local_performance():
    print("=" * 80)
    print("Executing Local Model Evaluation Pipeline (Task 1)")
    print("=" * 80)
    
    # Locate and load the dataset spreadsheet
    possible_paths = [
        Path("classification_task1_train.xlsx"),
        Path("data/classification_task1_train.xlsx"),
        Path("../data/classification_task1_train.xlsx")
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
            
    if data_path is None:
        print("Error: Dataset 'classification_task1_train.xlsx' not found.")
        sys.exit(1)
        
    # Load model state artifact
    try:
        loaded_model = my_model.load_model("best_model")
    except Exception as e:
        print(f"Error while loading model checkpoint: {e}")
        sys.exit(1)
        
    df = pd.read_excel(data_path)
    
    # Isolated validation split using the year 2020 as a proxy for the test set
    df_val = df[df['Year'] == 2020].copy()
    if df_val.empty:
        print("Warning: Validation split for Year 2020 is empty. Falling back to the last 20% of data.")
        df_val = df.iloc[int(len(df)*0.8):].copy()
        
    y_true = df_val['HAB'].to_numpy().astype(np.int64)
    
    # Drop target column to match evaluation environment conditions
    X_val_df = df_val.drop(columns=['HAB'])
    
    # Run vectorized inference via NumPy forward pass execution
    y_pred = my_model.predict(loaded_model, X_val_df)
    
    print("\n" + "-" * 50)
    print("CONFUSION MATRIX ANALYSIS")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    print(f" True Negatives  (No HAB correctly caught): {cm[0][0]}")
    print(f" False Positives (False Alarms generated)  : {cm[0][1]}")
    print(f" False Negatives (Missed HAB events!)      : {cm[1][0]}")
    print(f" True Positives  (HAB correctly caught)   : {cm[1][1]}")
    
    print("\n" + "-" * 50)
    print("CORE PERFORMANCE METRICS REPORT")
    print("-" * 50)
    # Target validation evaluation according to class notes guidelines
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f" -> Area Under the Curve (AUC Score): {auc:.4f}  (Main Evaluation Metric)")
    print(f" -> Balanced F1-Score               : {f1:.4f}  (Mitigates the Sham Accuracy)")
    
    print("\n" + "-" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=["No HAB (0)", "HAB (1)"], zero_division=0))
    print("=" * 80)

if __name__ == "__main__":
    evaluate_local_performance()