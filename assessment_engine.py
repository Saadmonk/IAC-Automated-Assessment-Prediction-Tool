from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from paper2.common import category_name, census_region, normalize_state
from paper2.config import (
    ARC_CATEGORY_NAMES,
    CURRENT_PRICE_REFERENCE_YEAR,
    STATE_ABBR,
    STAGE1_CATEGORICAL_FEATURES,
    STAGE2_CATEGORICAL_FEATURES_BY_SET,
    load_emission_factors,
)
from paper2.features import build_stage1_features, build_stage2_features

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
DATA_DIR = BASE / "data"

SYSTEM_CHECKLIST = {
    "Boilers & Steam System": ["Boilers & Steam System"],
    "Process Heating / Furnaces": ["Process Heating / Furnaces"],
    "Heat Recovery": ["Heat Recovery"],
    "Compressed Air System": ["Compressed Air System"],
    "Electric Motors & Drives": ["Electric Motors & Drives"],
    "Lighting (interior/exterior)": ["Lighting (interior/exterior)"],
    "HVAC / Space Conditioning": ["HVAC / Space Conditioning"],
    "Electrical Distribution": ["Electrical Distribution"],
    "Operations & Scheduling": ["Operations & Scheduling"],
    "Waste Management / Recycling": ["Waste Management / Recycling"],
    "Water Systems": ["Water Systems"],
    "Renewable / Alternative Energy": ["Renewable / Alternative Energy"],
}

SYSTEM_OPTIONS = list(SYSTEM_CHECKLIST.keys())
SECTOR_OPTIONS = [
    "Accommodation/Food Services",
    "Administrative Services",
    "Agriculture/Forestry/Fishing",
    "Arts/Entertainment",
    "Chemical/Plastics/Paper/Rubber",
    "Construction",
    "Educational Services",
    "Finance/Insurance",
    "Food/Beverage/Tobacco",
    "Health Care",
    "Information",
    "Metal/Machinery/Electronics",
    "Mining/Oil & Gas",
    "Other Services",
    "Professional Services",
    "Public Administration",
    "Real Estate",
    "Retail Trade",
    "Transportation",
    "Utilities",
    "Wholesale Trade",
]
STATE_OPTIONS = sorted(x for x in STATE_ABBR if len(x) == 2)

TOP_CATEGORY_BUFFER = 4
TOP_CANDIDATES_PER_CATEGORY = 8
TOP_SHADOW = 3
RECOMMENDATION_VALUE_CAP = 0.85
RECOMMENDATION_ENERGY_CAP = 0.85
PORTFOLIO_ENERGY_CAP = 0.90
PORTFOLIO_VALUE_CAP = 0.90
KWH_PER_MMBTU = 293.071
MMBTU_PER_KWH = 0.003412

_cache: dict[str, object] = {}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _load() -> dict[str, object]:
    if _cache.get("ready"):
        return _cache

    stage1_models = joblib.load(MODELS_DIR / "arc_category_models.joblib")
    stage2_models = joblib.load(MODELS_DIR / "facility_savings_models.joblib")
    arc_stats = pd.read_csv(MODELS_DIR / "arc_recommendation_statistics.csv")
    arc_meta = _load_arc_meta()
    prices_df = pd.read_csv(DATA_DIR / "external" / "state_year_energy_prices.csv")
    price_lookup = _build_price_lookup(prices_df)
    guardrails = {
        "arc": pd.read_csv(MODELS_DIR / "facility_guardrails.csv"),
        "meta": json.loads((MODELS_DIR / "facility_guardrails_meta.json").read_text(encoding="utf-8")),
    }
    stage2_defaults = stage2_models["current_value"]["artifacts"].medians
    ef = load_emission_factors()

    _cache.update(
        {
            "stage1_models": stage1_models,
            "stage2_models": stage2_models,
            "arc_stats": arc_stats,
            "arc_meta": arc_meta,
            "price_lookup": price_lookup,
            "guardrails": guardrails,
            "defaults": stage2_defaults,
            "weighted_emission_factor": float(ef.get("weighted", 59.91)),
            "ready": True,
        }
    )
    return _cache


def _load_arc_meta() -> pd.DataFrame:
    arc = pd.read_csv(DATA_DIR / "arc_codes.csv")
    arc.columns = [c.upper() for c in arc.columns]
    arc["ARC"] = arc["ARC_CODE"].astype(str).str.strip()
    arc["ARC_2DIGIT"] = pd.to_numeric(arc["ARC_2DIGIT"], errors="coerce")
    arc["CATEGORY_NAME"] = arc["CATEGORY_NAME"].fillna(arc["ARC_2DIGIT"].map(ARC_CATEGORY_NAMES)).fillna("Unknown")
    tags = arc.apply(
        lambda row: _infer_applicability_tags(
            row["ARC_2DIGIT"],
            str(row.get("DESCRIPTION", "")),
            str(row.get("SUBCATEGORY", "")),
        ),
        axis=1,
    )
    arc["APPLICABILITY_TAGS"] = tags.apply(lambda x: "|".join(x))
    arc["PRIMARY_TOOL_SYSTEM"] = tags.apply(lambda x: x[0] if x else "")
    return arc


def _build_price_lookup(prices_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    latest = prices_df.sort_values(["STATE", "FY_CLIPPED"]).groupby("STATE", as_index=False).tail(1).copy()
    price_cols = [
        "STATE_YEAR_ELECTRICITY_PRICE_MMBTU",
        "STATE_YEAR_NATURAL_GAS_PRICE_MMBTU",
        "STATE_YEAR_ELEC_GAS_RATIO",
        "CURRENT_ELECTRICITY_PRICE_MMBTU",
        "CURRENT_NATURAL_GAS_PRICE_MMBTU",
        "CURRENT_ELEC_GAS_RATIO",
        "CURRENT_PRICE_YEAR",
        "FY_CLIPPED",
    ]
    defaults = {col: float(pd.to_numeric(latest[col], errors="coerce").median()) for col in price_cols if col in latest.columns}
    lookup: dict[str, dict[str, float]] = {"__default__": defaults}
    for _, row in latest.iterrows():
        state = normalize_state(row["STATE"])
        lookup[state] = {}
        for col in price_cols:
            value = row[col] if col in row.index else np.nan
            lookup[state][col] = float(value) if pd.notna(value) else defaults.get(col, 0.0)
    return lookup


def _fill_positive(value: float | int | None, default: float) -> float:
    try:
        value = float(value) if value is not None else np.nan
    except Exception:
        value = np.nan
    if pd.isna(value) or value < 0:
        return float(default)
    return float(value)


def _facility_frame(facility: dict, cache: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    defaults = cache["defaults"]
    price_lookup = cache["price_lookup"]
    guard_meta = cache["guardrails"]["meta"]

    state = normalize_state(facility.get("state", "LA"))
    sector = facility.get("sector") or "Metal/Machinery/Electronics"
    price_ctx = price_lookup.get(state, price_lookup["__default__"])

    employees = _fill_positive(facility.get("employees"), defaults["EMPLOYEES"])
    plant_area = _fill_positive(facility.get("plant_area_sqft"), defaults["PLANT_AREA"])
    annual_elec_kwh = _fill_positive(facility.get("annual_elec_kwh"), defaults["ANNUAL_ELECTRIC_USAGE_KWH"])
    annual_gas_mmbtu = _fill_positive(facility.get("annual_gas_mmbtu"), defaults["ANNUAL_GAS_USAGE_MMBTU"])

    fy_value = int(price_ctx.get("CURRENT_PRICE_YEAR", CURRENT_PRICE_REFERENCE_YEAR) or CURRENT_PRICE_REFERENCE_YEAR)
    fy_norm = (fy_value - guard_meta["fy_min"]) / max(guard_meta["fy_max"] - guard_meta["fy_min"], 1)
    fy_norm = float(np.clip(fy_norm, 0, 1))

    row = {
        "ASSESS_ID": "INPUT_FACILITY",
        "STATE": state,
        "CENSUS_REGION": census_region(state),
        "SECTOR_NAME": sector,
        "FY": fy_value,
        "FY_NORM": fy_norm,
        "EMPLOYEES": employees,
        "PLANT_AREA": plant_area,
        "ANNUAL_ELECTRIC_USAGE_KWH": annual_elec_kwh,
        "ANNUAL_GAS_USAGE_MMBTU": annual_gas_mmbtu,
        "ANNUAL_ELECTRIC_COST": facility.get("annual_elec_cost"),
        "ANNUAL_GAS_COST": facility.get("annual_gas_cost"),
        "STATE_YEAR_ELECTRICITY_PRICE_MMBTU": price_ctx.get("STATE_YEAR_ELECTRICITY_PRICE_MMBTU", price_ctx.get("CURRENT_ELECTRICITY_PRICE_MMBTU", 0.0)),
        "STATE_YEAR_NATURAL_GAS_PRICE_MMBTU": price_ctx.get("STATE_YEAR_NATURAL_GAS_PRICE_MMBTU", price_ctx.get("CURRENT_NATURAL_GAS_PRICE_MMBTU", 0.0)),
        "STATE_YEAR_ELEC_GAS_RATIO": price_ctx.get("STATE_YEAR_ELEC_GAS_RATIO", 1.0),
        "CURRENT_ELECTRICITY_PRICE_MMBTU": price_ctx.get("CURRENT_ELECTRICITY_PRICE_MMBTU", 0.0),
        "CURRENT_NATURAL_GAS_PRICE_MMBTU": price_ctx.get("CURRENT_NATURAL_GAS_PRICE_MMBTU", 0.0),
        "CURRENT_ELEC_GAS_RATIO": price_ctx.get("CURRENT_ELEC_GAS_RATIO", 1.0),
    }
    frame = pd.DataFrame([row])
    context = _facility_scale_context(frame.iloc[0].to_dict(), facility)
    return frame, context


def _facility_scale_context(filled_row: dict, raw_facility: dict) -> dict[str, float | str]:
    elec_kwh = float(filled_row["ANNUAL_ELECTRIC_USAGE_KWH"])
    gas_mmbtu = float(filled_row["ANNUAL_GAS_USAGE_MMBTU"])
    total_energy = elec_kwh * MMBTU_PER_KWH + gas_mmbtu

    raw_elec_cost = raw_facility.get("annual_elec_cost")
    raw_gas_cost = raw_facility.get("annual_gas_cost")
    actual_spend = (float(raw_elec_cost) if raw_elec_cost not in (None, "") else 0.0) + (float(raw_gas_cost) if raw_gas_cost not in (None, "") else 0.0)

    current_elec_price_per_kwh = float(filled_row["CURRENT_ELECTRICITY_PRICE_MMBTU"]) / KWH_PER_MMBTU
    current_gas_price_per_mmbtu = float(filled_row["CURRENT_NATURAL_GAS_PRICE_MMBTU"])
    estimated_spend = elec_kwh * current_elec_price_per_kwh + gas_mmbtu * current_gas_price_per_mmbtu
    total_spend = actual_spend if actual_spend > 0 else estimated_spend
    spend_basis = "provided" if actual_spend > 0 else "estimated_from_state_prices"
    blended_rate = total_spend / total_energy if total_energy > 0 else 0.0
    return {
        "total_site_energy_mmbtu": max(total_energy, 1e-6),
        "annual_utility_spend_usd": max(total_spend, 1e-6),
        "annual_utility_spend_basis": spend_basis,
        "current_electric_rate_per_kwh": current_elec_price_per_kwh,
        "current_gas_rate_per_mmbtu": current_gas_price_per_mmbtu,
        "blended_rate_per_mmbtu": blended_rate,
    }


def _catboost_frame(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = X.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _predict_stage1(stage1_models: list[dict], facility_df: pd.DataFrame, calibrated: bool = True) -> pd.DataFrame:
    X = build_stage1_features(facility_df)
    rows = []
    for item in stage1_models:
        family = item.get("model_family", "lightgbm")
        if family == "catboost":
            raw = item["model"].predict_proba(_catboost_frame(X, [c for c in STAGE1_CATEGORICAL_FEATURES if c in X.columns]))[:, 1]
        else:
            raw = item["model"].predict_proba(X)[:, 1]
        calibrator = item.get("calibrator")
        pred = np.asarray(calibrator.predict(raw)) if calibrated and calibrator is not None else raw
        rows.append(pd.DataFrame({
            "ASSESS_ID": facility_df["ASSESS_ID"].astype(str),
            "label": item["label"],
            "name": item["name"],
            "probability": pred,
            "raw_probability": raw,
            "pos_rate": item["pos_rate"],
            "model_family": family,
        }))
    return pd.concat(rows, ignore_index=True)


def _predict_stage2(bundle: dict, df: pd.DataFrame, target_name: str) -> np.ndarray:
    target_bundle = bundle[target_name]
    artifacts = target_bundle["artifacts"]
    X, _ = build_stage2_features(
        df,
        feature_set=artifacts.feature_set,
        medians=artifacts.medians,
        sector_stats=artifacts.sector_stats,
        encoder_prior=artifacts.encoder_prior,
        encoder_map=artifacts.encoder_map,
        rate_artifacts=artifacts.rate_artifacts,
        winsor_bounds=artifacts.winsor_bounds,
    )
    family = target_bundle.get("model_family", "lightgbm")
    cat_cols = STAGE2_CATEGORICAL_FEATURES_BY_SET[artifacts.feature_set]
    if family == "catboost":
        pred_log = target_bundle["model"].predict(_catboost_frame(X, cat_cols))
    else:
        pred_log = target_bundle["model"].predict(X)
    return np.expm1(pred_log).clip(min=0)


def _infer_applicability_tags(arc_2digit: float | int | str | None, description: str, subcategory: str) -> list[str]:
    text = f"{description} {subcategory}".lower()
    tags: list[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if any(k in text for k in ("boiler", "steam", "condensate", "steam trap")):
        add("Boilers & Steam System")
    if any(k in text for k in ("furnace", "oven", "kiln", "dryer", "direct fired", "burner", "combustion", "process heating", "flue gas")):
        add("Process Heating / Furnaces")
    if any(k in text for k in ("heat recovery", "waste heat", "economizer", "heat exchanger", "recuperator")):
        add("Heat Recovery")
    if any(k in text for k in ("compressed air", "air compressor", "air leak", "pneumatic")):
        add("Compressed Air System")
    if any(k in text for k in ("motor", "drive", "pump", "fan", "blower", "vfd", "variable speed")):
        add("Electric Motors & Drives")
    if any(k in text for k in ("lighting", "lamp", "fixture", "led", "illumination")):
        add("Lighting (interior/exterior)")
    if any(k in text for k in ("hvac", "space conditioning", "air conditioning", "chiller", "refrigeration", "cooling tower", "building envelope", "window", "roof")):
        add("HVAC / Space Conditioning")
    if any(k in text for k in ("electrical distribution", "transformer", "power factor", "demand", "voltage")):
        add("Electrical Distribution")
    if any(k in text for k in ("schedule", "scheduling", "shutdown", "start-up", "startup", "operating practice", "load management")):
        add("Operations & Scheduling")
    if any(k in text for k in ("waste", "recycling", "raw material", "post generation")):
        add("Waste Management / Recycling")
    if any(k in text for k in ("water", "wastewater", "cooling water")):
        add("Water Systems")
    if any(k in text for k in ("renewable", "solar", "alternative energy", "chp", "cogeneration", "fuel switch")):
        add("Renewable / Alternative Energy")

    try:
        label = float(arc_2digit) if arc_2digit is not None else None
    except Exception:
        label = None
    if label is not None:
        if label == 2.3:
            add("Electrical Distribution")
        elif label == 2.6 and not tags:
            add("Operations & Scheduling")
        elif label == 2.9:
            add("Renewable / Alternative Energy")
        elif label in {3.5, 3.6, 3.8} and not tags:
            add("Waste Management / Recycling")
        elif label == 3.4 and not tags:
            add("Water Systems")
    return tags


def _arc_guard_row(arc_code: str, arc_2digit: float, cache: dict[str, object]) -> dict[str, float]:
    guard_df = cache["guardrails"]["arc"]
    meta = cache["guardrails"]["meta"]
    row = guard_df[guard_df["ARC"].astype(str) == str(arc_code)]
    if row.empty:
        row = guard_df[pd.to_numeric(guard_df["ARC_2DIGIT"], errors="coerce") == float(arc_2digit)]
    if row.empty:
        return {
            "energy_frac_q90": float(meta["global_energy_frac_q90"]),
            "value_frac_q90": float(meta["global_value_frac_q90"]),
            "energy_p25": 0.0,
            "energy_median": 0.0,
            "energy_p75": 0.0,
            "current_value_median": 0.0,
            "benchmark_dollar_median": 0.0,
            "payback_median": np.nan,
            "impl_rate_hist": np.nan,
        }
    series = row.iloc[0]
    return {
        key: float(series[key]) if pd.notna(series[key]) else 0.0
        for key in [
            "energy_frac_q90",
            "value_frac_q90",
            "energy_p25",
            "energy_median",
            "energy_p75",
            "current_value_median",
            "benchmark_dollar_median",
            "payback_median",
            "impl_rate_hist",
        ]
    }


def _facility_portfolio_caps(scale_context: dict, sector: str, cache: dict[str, object]) -> dict[str, float]:
    meta = cache["guardrails"]["meta"]
    sector_caps = meta.get("sector_facility_caps", {})
    sector_row = sector_caps.get(str(sector), {})
    energy_frac = float(sector_row.get("energy_frac_q90", meta.get("facility_energy_frac_q90", PORTFOLIO_ENERGY_CAP)))
    value_frac = float(sector_row.get("value_frac_q90", meta.get("facility_value_frac_q90", PORTFOLIO_VALUE_CAP)))
    energy_frac = min(max(energy_frac, 0.05), PORTFOLIO_ENERGY_CAP)
    value_frac = min(max(value_frac, 0.05), PORTFOLIO_VALUE_CAP)
    return {
        "energy_frac_cap": energy_frac,
        "value_frac_cap": value_frac,
        "energy_abs_cap": float(scale_context["total_site_energy_mmbtu"]) * energy_frac,
        "value_abs_cap": float(scale_context["annual_utility_spend_usd"]) * value_frac,
    }


def _apply_scale_guard(raw_energy: float, raw_value: float, context: dict, guard_row: dict) -> tuple[float, float, bool]:
    total_energy = float(context["total_site_energy_mmbtu"])
    total_spend = float(context["annual_utility_spend_usd"])
    energy_ratio_cap = min(float(guard_row.get("energy_frac_q90", 0.0) or 0.0), RECOMMENDATION_ENERGY_CAP) or RECOMMENDATION_ENERGY_CAP
    value_ratio_cap = min(float(guard_row.get("value_frac_q90", 0.0) or 0.0), RECOMMENDATION_VALUE_CAP) or RECOMMENDATION_VALUE_CAP
    energy_cap = total_energy * energy_ratio_cap
    value_cap = total_spend * value_ratio_cap
    capped_energy = min(float(raw_energy), float(energy_cap))
    capped_value = min(float(raw_value), float(value_cap))
    applied = (capped_energy + 1e-9 < raw_energy) or (capped_value + 1e-9 < raw_value)
    return max(capped_energy, 0.0), max(capped_value, 0.0), applied


def _candidate_pool(cat_label: float, cache: dict[str, object]) -> pd.DataFrame:
    arc_stats = cache["arc_stats"]
    arc_meta = cache["arc_meta"][["ARC", "DESCRIPTION", "SUBCATEGORY", "APPLICABILITY_TAGS", "PRIMARY_TOOL_SYSTEM"]]
    pool = arc_stats[arc_stats["ARC_2DIGIT"] == cat_label].copy()
    pool = pool.sort_values(["n_recs", "impl_rate", "median_current_value"], ascending=[False, False, False]).head(TOP_CANDIDATES_PER_CATEGORY)
    if pool.empty:
        return pool
    pool["ARC"] = pool["ARC"].astype(str)
    return pool.merge(arc_meta, on="ARC", how="left")


def _checklist_tags_from_text(value: str) -> list[str]:
    return [x for x in str(value).split("|") if x]


def _build_recommendation(candidate_row: pd.Series, cat_prob: float, selected_systems: set[str], scale_context: dict, cache: dict[str, object]) -> dict:
    arc_code = str(candidate_row["ARC"])
    cat_label = float(candidate_row["ARC_2DIGIT"])
    raw_energy = float(candidate_row["pred_energy"])
    raw_value = float(candidate_row["pred_value"])
    guard_row = _arc_guard_row(arc_code, cat_label, cache)
    energy_value, dollar_value, scale_guard_applied = _apply_scale_guard(raw_energy, raw_value, scale_context, guard_row)

    tags = _checklist_tags_from_text(str(candidate_row.get("APPLICABILITY_TAGS", "")))
    user_match = bool(selected_systems and set(tags).intersection(selected_systems))
    requires_system_filter = bool(selected_systems and tags and not user_match)
    savings_term = 0.5 * np.log1p(max(energy_value, 0)) + 0.5 * np.log1p(max(dollar_value, 0))
    composite = float(cat_prob) * float(candidate_row["impl_rate"]) * savings_term
    payback = float(guard_row.get("payback_median", np.nan))
    payback_out = round(payback, 1) if np.isfinite(payback) and 0 < payback <= 50 else None
    expected_energy = float(cat_prob) * energy_value
    expected_dollar = float(cat_prob) * dollar_value

    return {
        "arc_code": arc_code,
        "arc_system": category_name(cat_label),
        "arc_2digit": cat_label,
        "description": str(candidate_row.get("DESCRIPTION", "N/A")),
        "subcategory": str(candidate_row.get("SUBCATEGORY", "")),
        "p_category": round(float(cat_prob), 3),
        "system_selected": user_match,
        "impl_rate_hist": round(float(candidate_row["impl_rate"]), 3),
        "n_historical": int(candidate_row["n_recs"]),
        "composite_score": round(composite, 4),
        "energy_mmbtu_model": round(energy_value, 1),
        "dollar_model": round(dollar_value, 0),
        "energy_mmbtu_expected": round(expected_energy, 1),
        "dollar_expected": round(expected_dollar, 0),
        "energy_mmbtu_raw": round(raw_energy, 1),
        "dollar_model_raw": round(raw_value, 0),
        "energy_mmbtu_p25": round(float(guard_row.get("energy_p25", 0.0)), 1),
        "energy_mmbtu_median": round(float(guard_row.get("energy_median", 0.0)), 1),
        "energy_mmbtu_p75": round(float(guard_row.get("energy_p75", 0.0)), 1),
        "dollar_median_hist": round(float(candidate_row.get("median_benchmark_dollar", 0.0)), 0),
        "payback_est_yrs": payback_out,
        "applicability_tags": tags,
        "applicability_filter_applied": requires_system_filter,
        "shadow_system_needed": tags[0] if tags else str(candidate_row.get("PRIMARY_TOOL_SYSTEM", "")),
        "scale_guard_applied": scale_guard_applied,
    }


def run_assessment(facility: dict, top_n_categories: int = 6, top_n_per_cat: int = 3, n_shadow: int = TOP_SHADOW) -> dict:
    cache = _load()
    facility_df, scale_context = _facility_frame(facility, cache)
    selected_systems = set(facility.get("systems_present", []) or [])
    portfolio_caps = _facility_portfolio_caps(scale_context, str(facility_df.iloc[0]["SECTOR_NAME"]), cache)

    stage1_probs = _predict_stage1(cache["stage1_models"], facility_df, calibrated=True)
    top_labels = stage1_probs.sort_values("probability", ascending=False).head(min(len(stage1_probs), top_n_categories + TOP_CATEGORY_BUFFER)).copy()

    candidates: list[dict] = []
    for _, p_row in top_labels.iterrows():
        pool = _candidate_pool(float(p_row["label"]), cache)
        if pool.empty:
            continue
        candidate_input = pd.concat([facility_df] * len(pool), ignore_index=True)
        candidate_input["ARC"] = pool["ARC"].astype(str).to_list()
        candidate_input["ARC_2DIGIT"] = pd.to_numeric(pool["ARC_2DIGIT"], errors="coerce").to_list()
        candidate_input["PRIMARY_SOURCE_GROUP"] = pool["primary_source_group_mode"].fillna("unknown").astype(str).to_list()
        candidate_input["PRIMARY_FUEL_BUCKET"] = pool["primary_fuel_bucket_mode"].fillna("unknown").astype(str).to_list()
        pool = pool.reset_index(drop=True)
        pool["pred_energy"] = _predict_stage2(cache["stage2_models"], candidate_input, "technical_total")
        pool["pred_value"] = _predict_stage2(cache["stage2_models"], candidate_input, "current_value")
        for _, candidate_row in pool.iterrows():
            candidates.append(_build_recommendation(candidate_row, float(p_row["probability"]), selected_systems, scale_context, cache))

    if not candidates:
        return _json_safe({
            "recommendations": [],
            "shadow_recs": [],
            "portfolio": _portfolio_summary([], scale_context, portfolio_caps, cache),
            "facility_summary": _facility_summary(facility_df.iloc[0].to_dict(), facility, scale_context),
        })

    candidate_df = pd.DataFrame(candidates).sort_values("composite_score", ascending=False)
    candidate_df = candidate_df.drop_duplicates("arc_code", keep="first").reset_index(drop=True)
    main_df = candidate_df[~candidate_df["applicability_filter_applied"]].copy()
    shadow_df = candidate_df[candidate_df["applicability_filter_applied"]].copy()

    selected_main: list[dict] = []
    category_counts: dict[float, int] = {}
    used_categories: set[float] = set()
    cumulative_energy = 0.0
    cumulative_value = 0.0
    for row in main_df.sort_values(["system_selected", "composite_score"], ascending=[False, False]).to_dict("records"):
        cat = float(row["arc_2digit"])
        if category_counts.get(cat, 0) >= top_n_per_cat:
            continue
        if len(used_categories) >= top_n_categories and cat not in used_categories:
            continue
        next_energy = cumulative_energy + float(row["energy_mmbtu_model"])
        next_value = cumulative_value + float(row["dollar_model"])
        if selected_main and (next_energy > portfolio_caps["energy_abs_cap"] or next_value > portfolio_caps["value_abs_cap"]):
            continue
        used_categories.add(cat)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        selected_main.append(row)
        cumulative_energy = next_energy
        cumulative_value = next_value

    for idx, rec in enumerate(selected_main, start=1):
        rec["rank"] = idx

    shadow_recs = shadow_df.head(n_shadow).to_dict("records")
    for idx, rec in enumerate(shadow_recs, start=1):
        rec["shadow_rank"] = rec.get("shadow_rank") or idx
        rec["is_shadow"] = True

    return _json_safe({
        "recommendations": selected_main,
        "shadow_recs": shadow_recs,
        "portfolio": _portfolio_summary(selected_main, scale_context, portfolio_caps, cache),
        "facility_summary": _facility_summary(facility_df.iloc[0].to_dict(), facility, scale_context),
    })


def _portfolio_summary(recommendations: list[dict], scale_context: dict, portfolio_caps: dict, cache: dict[str, object]) -> dict:
    if not recommendations:
        return {
            "total_recommendations": 0,
            "total_energy_mmbtu": 0.0,
            "total_energy_mmbtu_potential": 0.0,
            "total_dollar_savings": 0.0,
            "total_dollar_savings_potential": 0.0,
            "total_co2_mt": 0.0,
            "total_co2_mt_potential": 0.0,
            "savings_pct_of_usage": 0.0,
            "savings_pct_of_usage_potential": 0.0,
            "savings_pct_of_bill": 0.0,
            "savings_pct_of_bill_potential": 0.0,
            "top_system": None,
            "annual_utility_spend_usd": round(float(scale_context["annual_utility_spend_usd"]), 0),
            "annual_utility_spend_basis": scale_context["annual_utility_spend_basis"],
            "portfolio_energy_cap_pct": round(float(portfolio_caps["energy_frac_cap"]) * 100.0, 1),
            "portfolio_value_cap_pct": round(float(portfolio_caps["value_frac_cap"]) * 100.0, 1),
            "portfolio_total_style": "expected_value",
            "probability_note": "Expected totals weight each recommendation by the predicted ARC category probability.",
        }
    raw_total_energy_potential = float(sum(x["energy_mmbtu_model"] for x in recommendations))
    raw_total_value_potential = float(sum(x["dollar_model"] for x in recommendations))
    raw_total_energy_expected = float(sum(x.get("energy_mmbtu_expected", 0.0) for x in recommendations))
    raw_total_value_expected = float(sum(x.get("dollar_expected", 0.0) for x in recommendations))

    total_energy_potential = min(raw_total_energy_potential, float(portfolio_caps["energy_abs_cap"]))
    total_value_potential = min(raw_total_value_potential, float(portfolio_caps["value_abs_cap"]))
    total_energy = min(raw_total_energy_expected, float(portfolio_caps["energy_abs_cap"]))
    total_value = min(raw_total_value_expected, float(portfolio_caps["value_abs_cap"]))

    total_co2 = total_energy * float(cache["weighted_emission_factor"]) / 1000.0
    total_co2_potential = total_energy_potential * float(cache["weighted_emission_factor"]) / 1000.0
    usage_pct = 100.0 * total_energy / max(float(scale_context["total_site_energy_mmbtu"]), 1e-6)
    usage_pct_potential = 100.0 * total_energy_potential / max(float(scale_context["total_site_energy_mmbtu"]), 1e-6)
    bill_pct = 100.0 * total_value / max(float(scale_context["annual_utility_spend_usd"]), 1e-6)
    bill_pct_potential = 100.0 * total_value_potential / max(float(scale_context["annual_utility_spend_usd"]), 1e-6)
    return {
        "total_recommendations": len(recommendations),
        "total_energy_mmbtu": round(total_energy, 1),
        "total_energy_mmbtu_potential": round(total_energy_potential, 1),
        "total_dollar_savings": round(total_value, 0),
        "total_dollar_savings_potential": round(total_value_potential, 0),
        "total_co2_mt": round(total_co2, 2),
        "total_co2_mt_potential": round(total_co2_potential, 2),
        "savings_pct_of_usage": round(usage_pct, 1),
        "savings_pct_of_usage_potential": round(usage_pct_potential, 1),
        "savings_pct_of_bill": round(bill_pct, 1),
        "savings_pct_of_bill_potential": round(bill_pct_potential, 1),
        "top_system": recommendations[0]["arc_system"],
        "annual_utility_spend_usd": round(float(scale_context["annual_utility_spend_usd"]), 0),
        "annual_utility_spend_basis": scale_context["annual_utility_spend_basis"],
        "portfolio_energy_cap_pct": round(float(portfolio_caps["energy_frac_cap"]) * 100.0, 1),
        "portfolio_value_cap_pct": round(float(portfolio_caps["value_frac_cap"]) * 100.0, 1),
        "portfolio_total_style": "expected_value",
        "probability_note": "Expected totals weight each recommendation by the predicted ARC category probability.",
        "portfolio_guard_applied": (
            (total_energy_potential + 1e-9 < raw_total_energy_potential)
            or (total_value_potential + 1e-9 < raw_total_value_potential)
            or (total_energy + 1e-9 < raw_total_energy_expected)
            or (total_value + 1e-9 < raw_total_value_expected)
        ),
    }


def _facility_summary(filled_row: dict, raw_facility: dict, scale_context: dict) -> dict:
    return {
        "sector": filled_row["SECTOR_NAME"],
        "state": filled_row["STATE"],
        "employees": int(round(float(filled_row["EMPLOYEES"]))),
        "plant_area_sqft": int(round(float(filled_row["PLANT_AREA"]))),
        "annual_elec_kwh": round(float(filled_row["ANNUAL_ELECTRIC_USAGE_KWH"]), 1),
        "annual_gas_mmbtu": round(float(filled_row["ANNUAL_GAS_USAGE_MMBTU"]), 1),
        "annual_elec_cost": raw_facility.get("annual_elec_cost"),
        "annual_gas_cost": raw_facility.get("annual_gas_cost"),
        "annual_utility_spend_usd": round(float(scale_context["annual_utility_spend_usd"]), 0),
        "annual_utility_spend_basis": scale_context["annual_utility_spend_basis"],
        "systems_present": raw_facility.get("systems_present", []) or [],
    }
