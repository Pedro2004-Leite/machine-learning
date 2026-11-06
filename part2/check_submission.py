"""Our own submission check for Part II (the supplied validator is truncated).

Usage: python check_submission.py TASK SUBMISSION_DIR RELEASED_DATA [--allow-missing-declarations]
  TASK is 1 (HAB, labels {0,1}) or 2 (PN, labels {0,1,2,3}).

Does what validate_submission_2.py does (structure, forbidden constructs, import,
load_model) and then what it forgot: calls predict on a few released rows and
checks shape, finiteness, label set, no in-place modification, and that it works
when Python is started from another directory.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from model_artifact_policy import (
    validate_explicit_transformation_sources,
    validate_explicit_transformations,
)

TARGETS = {1: "HAB", 2: "PN"}
LABELS = {1: {0, 1}, 2: {0, 1, 2, 3}}


def read_table(path):
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def import_module(path):
    module_dir = str(path.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("student_my_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_pdf(path):
    if not path.is_file() or path.stat().st_size < 5:
        return False
    with path.open("rb") as handle:
        return b"%PDF-" in handle.read(1024)


def check(task, submission_dir, data_path, require_declarations=True, n_rows=20):
    best = submission_dir / "best_model"
    train = submission_dir / "training_code"
    if not best.is_dir():
        raise ValueError("missing directory: best_model")
    missing = [n for n in ["my_model.py", "model.pkl", "utils.py"] if not (best / n).is_file()]
    if missing:
        raise ValueError(f"missing files in best_model: {missing}")
    if not (train / "train.py").is_file():
        raise ValueError("missing file: training_code/train.py")
    sources = sorted(p for d in (best, train) for p in d.rglob("*.py"))
    validate_explicit_transformation_sources(sources)

    if require_declarations:
        decl = submission_dir / "declarations"
        for name in ["group_declaration_of_honor.pdf", "student_1_ai_declaration.pdf",
                     "student_2_ai_declaration.pdf"]:
            if not is_pdf(decl / name):
                raise ValueError(f"declarations/{name} missing or not a valid PDF")

    module = import_module(best / "my_model.py")
    for name in ["load_model", "predict"]:
        if not callable(getattr(module, name, None)):
            raise ValueError(f"my_model.py must define callable {name}")
    model = module.load_model(best / "model.pkl")
    validate_explicit_transformations(model)

    raw = read_table(data_path)
    target = TARGETS[task]
    if target not in raw:
        raise ValueError(f"released data has no {target} column (wrong TASK?)")
    # a spread of rows, including ones with missing values
    nan_rows = raw.index[raw.drop(columns=[target]).isna().any(axis=1)].tolist()
    picks = sorted(set(np.linspace(0, len(raw) - 1, n_rows, dtype=int).tolist() + nan_rows[:n_rows]))

    checks = []
    start_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as other_dir:
        os.chdir(other_dir)  # predict must not depend on the working directory
        try:
            for i in picks:
                day = raw.iloc[[i]].drop(columns=[target]).copy()
                before = day.copy(deep=True)
                pred = np.asarray(module.predict(model, day), dtype=float)
                if pred.shape != (1,) or not np.isfinite(pred).all():
                    raise ValueError(f"row {i}: predict must return one finite value, got {pred!r}")
                if int(pred[0]) != pred[0] or int(pred[0]) not in LABELS[task]:
                    raise ValueError(f"row {i}: prediction {pred[0]} not in {sorted(LABELS[task])}")
                if not day.equals(before):
                    raise ValueError(f"row {i}: predict modified the input DataFrame")
                checks.append(int(pred[0]))
        finally:
            os.chdir(start_cwd)
    counts = {int(k): checks.count(k) for k in sorted(set(checks))}
    return {"status": "PASS", "task": task, "group_id": submission_dir.name,
            "rows_checked": len(picks), "predicted_label_counts": counts,
            "rows_with_missing_values_checked": len([i for i in picks if i in set(nan_rows)])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int, choices=[1, 2])
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("released_data", type=Path)
    parser.add_argument("--allow-missing-declarations", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.task, args.submission_dir, args.released_data,
                           not args.allow_missing_declarations), indent=2))


if __name__ == "__main__":
    main()
