from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import safe_log1p, safe_ratio, to_category_columns
from .config import RATE_CLIP_QUANTILES, STAGE1_CATEGORICAL_FEATURES, STAGE2_CATEGORICAL_FEATURES_BY_SET, STAGE2_FEATURE_SETS


@dataclass
class Stage2FeatureArtifacts:
    medians: dict
    sector_stats: pd.DataFrame
    encoder_prior: float
    encoder_map: pd.DataFrame
    rate_artifacts: dict
    winsor_bounds: dict
    feature_set: str


def fill_with_medians(df: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    columns = [
        "EMPLOYEES",
        "PLANT_AREA",
        "ANNUAL_ELECTRIC_USAGE_KWH",
        "ANNUAL_GAS_USAGE_MMBTU",
        "ANNUAL_ELECTRIC_COST",
        "ANNUAL_GAS_COST",
        "STATE_YEAR_ELECTRICITY_PRICE_MMBTU",
        "STATE_YEAR_NATURAL_GAS_PRICE_MMBTU",
        "STATE_YEAR_ELEC_GAS_RATIO",
        "CURRENT_ELECTRICITY_PRICE_MMBTU",
        "CURRENT_NATURAL_GAS_PRICE_MMBTU",
        "CURRENT_ELEC_GAS_RATIO",
    ]
    fitted = {}
    for col in columns:
        median = medians[col] if medians and col in medians else pd.to_numeric(out[col], errors="coerce").median()
        fitted[col] = float(median if pd.notna(median) else 0.0)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fitted[col]).clip(lower=0)
    return out, fitted


def build_stage1_features(df: pd.DataFrame) -> pd.DataFrame:
    out, _ = fill_with_medians(df)
    feat = pd.DataFrame({
        "log_employees": safe_log1p(out["EMPLOYEES"]),
        "log_plant_area": safe_log1p(out["PLANT_AREA"]),
        "log_annual_elec_kwh": safe_log1p(out["ANNUAL_ELECTRIC_USAGE_KWH"]),
        "log_annual_gas_mmbtu": safe_log1p(out["ANNUAL_GAS_USAGE_MMBTU"]),
        "fy_norm": pd.to_numeric(df["FY_NORM"], errors="coerce").fillna(0),
        "sector_name": df["SECTOR_NAME"].fillna("Unknown").astype(str),
        "state": df["STATE"].fillna("UNK").astype(str),
        "census_region": df["CENSUS_REGION"].fillna("Unknown").astype(str),
        "state_year_electricity_price_mmbtu": pd.to_numeric(out["STATE_YEAR_ELECTRICITY_PRICE_MMBTU"], errors="coerce").fillna(0),
        "state_year_natural_gas_price_mmbtu": pd.to_numeric(out["STATE_YEAR_NATURAL_GAS_PRICE_MMBTU"], errors="coerce").fillna(0),
        "state_year_elec_gas_ratio": pd.to_numeric(out["STATE_YEAR_ELEC_GAS_RATIO"], errors="coerce").fillna(0),
    })
    return to_category_columns(feat, STAGE1_CATEGORICAL_FEATURES)


def build_stage2_features(
    df: pd.DataFrame,
    feature_set: str = "source_aware",
    medians: dict | None = None,
    sector_stats: pd.DataFrame | None = None,
    encoder_prior: float | None = None,
    encoder_map: pd.DataFrame | None = None,
    rate_artifacts: dict | None = None,
    winsor_bounds: dict | None = None,
) -> tuple[pd.DataFrame, Stage2FeatureArtifacts]:
    if feature_set not in STAGE2_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}")
    df = df.reset_index(drop=True)
    missing_flags = _missing_flags(df)
    filled, fitted_medians = fill_with_medians(df, medians)
    base = _build_stage2_base_numeric(filled)
    base["sector_name"] = df["SECTOR_NAME"].fillna("Unknown").astype(str)
    base["state"] = df["STATE"].fillna("UNK").astype(str)
    base["census_region"] = df["CENSUS_REGION"].fillna("Unknown").astype(str)
    base["arc"] = df["ARC"].fillna("").astype(str)
    base["arc_2digit"] = df["ARC_2DIGIT"].astype(str)
    base["fy_norm"] = pd.to_numeric(df["FY_NORM"], errors="coerce").fillna(0)
    base["state_year_electricity_price_mmbtu"] = pd.to_numeric(filled["STATE_YEAR_ELECTRICITY_PRICE_MMBTU"], errors="coerce").fillna(0)
    base["state_year_natural_gas_price_mmbtu"] = pd.to_numeric(filled["STATE_YEAR_NATURAL_GAS_PRICE_MMBTU"], errors="coerce").fillna(0)
    base["state_year_elec_gas_ratio"] = pd.to_numeric(filled["STATE_YEAR_ELEC_GAS_RATIO"], errors="coerce").fillna(0)
    base["current_electricity_price_mmbtu"] = pd.to_numeric(filled["CURRENT_ELECTRICITY_PRICE_MMBTU"], errors="coerce").fillna(0)
    base["current_natural_gas_price_mmbtu"] = pd.to_numeric(filled["CURRENT_NATURAL_GAS_PRICE_MMBTU"], errors="coerce").fillna(0)
    base["current_elec_gas_ratio"] = pd.to_numeric(filled["CURRENT_ELEC_GAS_RATIO"], errors="coerce").fillna(0)

    if sector_stats is None:
        sector_stats = pd.DataFrame({"SECTOR_NAME": [], "ANNUAL_ELECTRIC_USAGE_KWH_mean": [], "ANNUAL_ELECTRIC_USAGE_KWH_std": []})
    out = base.merge(sector_stats, left_on="sector_name", right_on="SECTOR_NAME", how="left")
    if "SECTOR_NAME" in out.columns:
        out.drop(columns=["SECTOR_NAME"], inplace=True)

    z_pairs = [
        ("ANNUAL_ELECTRIC_USAGE_KWH", "z_annual_elec_kwh"),
        ("ANNUAL_GAS_USAGE_MMBTU", "z_annual_gas_mmbtu"),
        ("total_energy", "z_total_energy"),
        ("kwh_per_employee", "z_kwh_per_employee"),
        ("kwh_per_sqft", "z_kwh_per_sqft"),
        ("total_energy_per_employee", "z_total_energy_per_employee"),
        ("total_energy_per_sqft", "z_total_energy_per_sqft"),
    ]
    for raw_col, z_col in z_pairs:
        mean_col = f"{raw_col}_mean"
        std_col = f"{raw_col}_std"
        mean_vals = pd.to_numeric(out[mean_col], errors="coerce") if mean_col in out.columns else 0
        std_vals = pd.to_numeric(out[std_col], errors="coerce").replace(0, np.nan) if std_col in out.columns else np.nan
        out[z_col] = ((pd.to_numeric(out[raw_col], errors="coerce") - mean_vals) / std_vals).replace([np.inf, -np.inf], np.nan).fillna(0)

    if encoder_map is None:
        encoder_prior = 0.0 if encoder_prior is None else encoder_prior
        encoder_map = pd.DataFrame({"ARC": [], "encoded": []})
    enc = encoder_map.rename(columns={"ARC": "arc", "encoded": "arc_target_encode"})
    out = out.merge(enc[["arc", "arc_target_encode"]], on="arc", how="left")
    out["arc_target_encode"] = out["arc_target_encode"].fillna(float(encoder_prior or 0.0))

    if rate_artifacts is None:
        rate_artifacts = {
            "elec_clip": [0.0, 1.0],
            "gas_clip": [0.0, 20.0],
            "sector_rate_stats": pd.DataFrame({"SECTOR_NAME": [], "elec_rate_sector_median": [], "gas_rate_sector_median": []}),
            "global_elec_rate_per_kwh": 0.06,
            "global_gas_rate_per_mmbtu": 5.5,
        }
    rates = _build_rate_features(df, rate_artifacts)
    out = pd.concat([out.reset_index(drop=True), missing_flags.reset_index(drop=True), rates.reset_index(drop=True)], axis=1)
    out["primary_source_group"] = df.get("PRIMARY_SOURCE_GROUP", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    out["primary_fuel_bucket"] = df.get("PRIMARY_FUEL_BUCKET", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)

    if winsor_bounds is None:
        winsor_bounds = {}
    out = apply_numeric_winsor_bounds(out, winsor_bounds)

    selected = STAGE2_FEATURE_SETS[feature_set]
    final = out[selected].copy()
    final = to_category_columns(final, STAGE2_CATEGORICAL_FEATURES_BY_SET[feature_set])
    artifacts = Stage2FeatureArtifacts(
        medians=fitted_medians,
        sector_stats=sector_stats,
        encoder_prior=float(encoder_prior or 0.0),
        encoder_map=encoder_map,
        rate_artifacts=rate_artifacts,
        winsor_bounds=winsor_bounds,
        feature_set=feature_set,
    )
    return final, artifacts


def _build_stage2_base_numeric(df: pd.DataFrame) -> pd.DataFrame:
    total_energy = df["ANNUAL_ELECTRIC_USAGE_KWH"] * 0.003412 + df["ANNUAL_GAS_USAGE_MMBTU"]
    kwh_per_employee = safe_ratio(df["ANNUAL_ELECTRIC_USAGE_KWH"], df["EMPLOYEES"])
    kwh_per_sqft = safe_ratio(df["ANNUAL_ELECTRIC_USAGE_KWH"], df["PLANT_AREA"])
    mmbtu_per_employee = safe_ratio(df["ANNUAL_GAS_USAGE_MMBTU"], df["EMPLOYEES"])
    total_energy_per_employee = safe_ratio(total_energy, df["EMPLOYEES"])
    total_energy_per_sqft = safe_ratio(total_energy, df["PLANT_AREA"])
    gas_fraction = safe_ratio(df["ANNUAL_GAS_USAGE_MMBTU"], total_energy)
    electric_share = safe_ratio(df["ANNUAL_ELECTRIC_USAGE_KWH"] * 0.003412, total_energy).fillna(0).clip(lower=0, upper=1)
    thermal_share = safe_ratio(df["ANNUAL_GAS_USAGE_MMBTU"], total_energy).fillna(0).clip(lower=0, upper=1)
    return pd.DataFrame({
        "ANNUAL_ELECTRIC_USAGE_KWH": df["ANNUAL_ELECTRIC_USAGE_KWH"],
        "ANNUAL_GAS_USAGE_MMBTU": df["ANNUAL_GAS_USAGE_MMBTU"],
        "total_energy": total_energy.fillna(0),
        "kwh_per_employee": kwh_per_employee.fillna(0),
        "kwh_per_sqft": kwh_per_sqft.fillna(0),
        "mmbtu_per_employee": mmbtu_per_employee.fillna(0),
        "total_energy_per_employee": total_energy_per_employee.fillna(0),
        "total_energy_per_sqft": total_energy_per_sqft.fillna(0),
        "gas_fraction_total_energy": gas_fraction.fillna(0).clip(lower=0, upper=1),
        "log_employees": safe_log1p(df["EMPLOYEES"]),
        "log_plant_area": safe_log1p(df["PLANT_AREA"]),
        "log_annual_elec_kwh": safe_log1p(df["ANNUAL_ELECTRIC_USAGE_KWH"]),
        "log_annual_gas_mmbtu": safe_log1p(df["ANNUAL_GAS_USAGE_MMBTU"]),
        "log_kwh_per_employee": safe_log1p(kwh_per_employee.fillna(0)),
        "log_kwh_per_sqft": safe_log1p(kwh_per_sqft.fillna(0)),
        "log_mmbtu_per_employee": safe_log1p(mmbtu_per_employee.fillna(0)),
        "log_total_energy_per_employee": safe_log1p(total_energy_per_employee.fillna(0)),
        "log_total_energy_per_sqft": safe_log1p(total_energy_per_sqft.fillna(0)),
        "log_total_energy": safe_log1p(total_energy.fillna(0)),
        "electric_share_total_energy": electric_share,
        "thermal_share_total_energy": thermal_share,
    })


def _missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "plant_area_missing": pd.to_numeric(df["PLANT_AREA"], errors="coerce").isna().astype(int),
        "annual_gas_usage_missing": pd.to_numeric(df["ANNUAL_GAS_USAGE_MMBTU"], errors="coerce").isna().astype(int),
        "annual_gas_cost_missing": pd.to_numeric(df["ANNUAL_GAS_COST"], errors="coerce").isna().astype(int),
        "annual_electric_cost_missing": pd.to_numeric(df["ANNUAL_ELECTRIC_COST"], errors="coerce").isna().astype(int),
    })


def _build_rate_features(df: pd.DataFrame, rate_artifacts: dict) -> pd.DataFrame:
    elec_rate = safe_ratio(df["ANNUAL_ELECTRIC_COST"], df["ANNUAL_ELECTRIC_USAGE_KWH"])
    gas_rate = safe_ratio(df["ANNUAL_GAS_COST"], df["ANNUAL_GAS_USAGE_MMBTU"])
    sector_rates = rate_artifacts.get("sector_rate_stats", pd.DataFrame({"SECTOR_NAME": [], "elec_rate_sector_median": [], "gas_rate_sector_median": []}))
    temp = pd.DataFrame({
        "SECTOR_NAME": df["SECTOR_NAME"].fillna("Unknown").astype(str),
        "elec_rate_raw": elec_rate,
        "gas_rate_raw": gas_rate,
    }).merge(sector_rates, on="SECTOR_NAME", how="left")
    elec_lo, elec_hi = rate_artifacts.get("elec_clip", [0.0, 1.0])
    gas_lo, gas_hi = rate_artifacts.get("gas_clip", [0.0, 20.0])
    elec_clipped = temp["elec_rate_raw"].clip(lower=elec_lo, upper=elec_hi)
    gas_clipped = temp["gas_rate_raw"].clip(lower=gas_lo, upper=gas_hi)
    elec_filled = elec_clipped.fillna(temp.get("elec_rate_sector_median")).fillna(rate_artifacts.get("global_elec_rate_per_kwh", 0.06))
    gas_filled = gas_clipped.fillna(temp.get("gas_rate_sector_median")).fillna(rate_artifacts.get("global_gas_rate_per_mmbtu", 5.5))
    return pd.DataFrame({
        "elec_rate_per_kwh_filled": elec_filled.clip(lower=0),
        "gas_rate_per_mmbtu_filled": gas_filled.clip(lower=0),
    })


def apply_numeric_winsor_bounds(df: pd.DataFrame, bounds: dict | None) -> pd.DataFrame:
    if not bounds:
        return df
    out = df.copy()
    for col, limits in bounds.items():
        if col not in out.columns:
            continue
        lo, hi = limits
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=float(lo), upper=float(hi))
    return out


def _clip_bounds(series: pd.Series) -> tuple[float, float]:
    if series.empty:
        return 0.0, 0.0
    lo, hi = series.quantile(RATE_CLIP_QUANTILES).tolist()
    lo = float(lo if pd.notna(lo) else 0.0)
    hi = float(hi if pd.notna(hi) else lo)
    if hi < lo:
        hi = lo
    return lo, hi

