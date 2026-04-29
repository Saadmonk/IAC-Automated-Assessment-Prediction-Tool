from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import ARC_CATEGORY_NAMES, CENSUS_REGION_BY_STATE


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / den.astype(float).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def safe_log1p(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    vals = pd.Series(series, copy=False).astype(float).clip(lower=0)
    return np.log1p(vals)


def normalize_state(value: object) -> str:
    text = str(value).strip().upper()
    return text if len(text) == 2 else "UNK"


def census_region(state: object) -> str:
    return CENSUS_REGION_BY_STATE.get(normalize_state(state), "Unknown")


def category_name(label: float) -> str:
    return ARC_CATEGORY_NAMES.get(label, f"Category {label}")


def to_category_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out

