"""Task 1 training entry point. PLACEHOLDER: writes the dummy model.

The real version must reproduce the full workflow from the released raw data and
save best_model/model.pkl. Run from anywhere: python training_code/train.py
"""

import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model" / "model.pkl"


def main():
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump({"constant_class": 0}, handle)


if __name__ == "__main__":
    main()
