from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import pandas as pd


LOGGER = logging.getLogger(__name__)


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_data(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = ensure_parent_dir(path)
    df.to_csv(path, index=index)
    LOGGER.info("Saved data to %s", path)


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def save_model(model: Any, path: str | Path) -> None:
    path = ensure_parent_dir(path)
    joblib.dump(model, path)
    LOGGER.info("Saved model to %s", path)


def load_model(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_json(obj: Any, path: str | Path) -> None:
    path = ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_files(paths: Iterable[str | Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def log_report(df: pd.DataFrame, report_path: str | Path = "reports/data_report.txt") -> None:
    report_path = ensure_parent_dir(report_path)

    duplicates = int(df.duplicated().sum())
    missing = df.isna().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== DATASET REPORT ===\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Duplicates: {duplicates}\n")
        f.write(f"Columns with missing values: {len(missing_cols)}\n\n")

        if len(missing_cols) > 0:
            f.write("--- Missing values detail ---\n")
            f.write(missing_cols.to_string())
            f.write("\n\n")

        f.write("--- Columns ---\n")
        f.write(", ".join(df.columns.tolist()) + "\n")