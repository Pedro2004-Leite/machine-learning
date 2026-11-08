import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path("best_model").resolve()))

try:
    import my_model
except ModuleNotFoundError:
    print("Error: Could not find 'best_model/my_model.py'. Check directory structure.")
    sys.exit(1)

def run_gpu_workflow_test():
    print("=" * 80)
    print("Stage 1: Launching neural network training script...")
    print("=" * 80)
    
    try:
        import training_code.train as train_script
        train_script.main()
    except Exception as e:
        print(f"Error during execution of train.py: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Stage 2: Verifying artifact serialization and delivery...")
    print("=" * 80)
    
    try:
        loaded_state = my_model.load_model("best_model")
        print("Success: model.pkl successfully serialized and loaded.")
    except Exception as e:
        print(f"Error while unpickling model.pkl: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 80)
    print("Stage 3: Validating prediction constraints and shapes...")
    print("=" * 80)
    
    # Robust multi-path lookup for the training spreadsheet to mock test inference
    possible_paths = [
        Path("classification_task1_train.xlsx"),
        Path("../data/classification_task1_train.xlsx"),
        Path("../classification_task1_train.xlsx"),
        Path("../../classification_task1_train.xlsx")
    ]
    
    data_path = None
    for p in possible_paths:
        if p.exists():
            data_path = p
            break
            
    if data_path is None:
        print("Error: Could not find 'classification_task1_train.xlsx' in any standard location.")
        print("Please ensure the spreadsheet is present in your project directory.")
        sys.exit(1)
        
    print(f"Loading validation slice from: {data_path}")
    df = pd.read_excel(data_path)
    
    # Filter specific year row instances and discard target column
    df_test = df[df['Year'] == 2020].copy()
    if 'HAB' in df_test.columns:
        df_test = df_test.drop(columns=['HAB'])
        
    print(f"Evaluation frame created with {len(df_test)} instances.")
    
    try:
        predictions = my_model.predict(loaded_state, df_test)
        
        print("\nChecking submission formatting rules:")
        print(f" -> Output type: {type(predictions)} (Expected: <class 'numpy.ndarray'>)")
        print(f" -> Array shape: {predictions.shape} (Expected: ({len(df_test)},))")
        print(f" -> Unique categories found: {np.unique(predictions)} (Expected: subset of {{0, 1}})")
        
        if isinstance(predictions, np.ndarray) and predictions.ndim == 1 and len(predictions) == len(df_test):
            print("\nSuccess: Array formatting requirements completely satisfied for Task 1.")
        else:
            print("\nWarning: Output matrix configuration verification failed.")
            
    except Exception as e:
        print(f"Critical error during predict function execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_gpu_workflow_test()