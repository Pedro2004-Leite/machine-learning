import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_exploratory_analysis_task1(data_path: str or Path):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset target not found: {path}")
        
    # Load dataset
    df = pd.read_excel(path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    
    print("=" * 80)
    print(f"Task 1 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("=" * 80)
    
    # 1. Target Binary Class Balance Analysis
    target_col = "HAB"
    if target_col in df.columns:
        print("\n[Target Distribution - HAB Binary Events]")
        counts = df[target_col].value_counts().sort_index()
        percentages = df[target_col].value_counts(normalize=True).sort_index() * 100
        for cls in counts.index:
            label = "No Occurrence (0)" if cls == 0 else "Occurrence (1)"
            print(f" {label}: {counts[cls]} samples ({percentages[cls]:.2f}%)")
            
        # Chart 1: Binary Target Distribution with Value Labels
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x=target_col, hue=target_col, palette="coolwarm", legend=False, ax=ax)
        
        # Add the exact counts on top of each bar automatically
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=10, weight='bold')
            
        plt.title("HAB Event Occurrence Distribution (Task 1)")
        plt.xlabel("HAB Target (0: No, 1: Yes)")
        plt.ylabel("Sample Count")
        
        # Adjust y-axis limit slightly to give breathing room for the text labels
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
        
        plt.tight_layout()
        plt.savefig("eda_task1_target_distribution.png")
        plt.close()

        
    # 2. Missing Values and CHL Analysis
    missing_series = df.isna().sum()
    missing_features = missing_series[missing_series > 0].sort_values(ascending=False)
    if not missing_features.empty:
        print("\n[Missing Features Percentage Report]")
        for col in missing_features.index:
            pct = (missing_features[col] / len(df)) * 100
            print(f" {col}: {missing_features[col]} null entries ({pct:.2f}%)")
            
        # Chart 2: Missing Data Pattern Heatmap
        plt.figure(figsize=(8, 4))
        sns.heatmap(df[["CHL", "CHL_uncertainty"]].isna(), cbar=False, yticklabels=False, cmap="mako")
        plt.title("Missing Value Matrix Heatmap for Chlorophyll Columns")
        plt.tight_layout()
        plt.savefig("eda_task1_missing_heatmap.png")
        plt.close()
        
    # 3. Environmental Feature Profiles by Target Class
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    base_vars = ["SST", "UI", "CHL"]
    available_base = [v for v in base_vars if v in numerical_cols]
    
    if available_base and target_col in df.columns:
        # Chart 3: Distribution Boxplots grouped by HAB state
        plt.figure(figsize=(12, 4))
        for i, var in enumerate(available_base, 1):
            plt.subplot(1, len(available_base), i)
            sns.boxplot(data=df, x=target_col, y=var, hue=target_col, palette="Set2", legend=False)
            plt.title(f"{var} Distribution by HAB Status")
        plt.tight_layout()
        plt.savefig("eda_task1_features_boxplot.png")
        plt.close()
        
    print("\n" + "=" * 80)
    print("Task 1 EDA completed successfully. Diagnostic plots exported to workspace.")
    print("=" * 80)

if __name__ == "__main__":
    # Fallback to local paths based on the structure validated in prior stages
    target_file = "classification_task1_train.xlsx"
    if not Path(target_file).exists():
        target_file = "../data/classification_task1_train.xlsx"
        
    run_exploratory_analysis_task1(target_file)