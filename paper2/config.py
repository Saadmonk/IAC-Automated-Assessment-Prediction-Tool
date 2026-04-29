from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CURRENT_PRICE_REFERENCE_YEAR = 2024
RATE_CLIP_QUANTILES = (0.01, 0.99)

ARC_CATEGORY_NAMES = {
    2.1: "Combustion Systems",
    2.2: "Thermal Systems",
    2.3: "Electrical Power",
    2.4: "Motor Systems",
    2.5: "Industrial Design",
    2.6: "Operations",
    2.7: "Building & Grounds",
    2.8: "Ancillary Costs",
    2.9: "Alternative Energy",
    3.1: "WM - Operations",
    3.2: "WM - Equipment",
    3.3: "WM - Post Generation Treatment",
    3.4: "WM - Water Use",
    3.5: "WM - Recycling",
    3.6: "WM - Waste Disposal",
    3.7: "WM - Maintenance",
    3.8: "WM - Raw Materials",
    4.1: "Productivity - Manufacturing",
    4.2: "Productivity - Purchasing",
    4.3: "Productivity - Inventory",
    4.4: "Productivity - Labor",
    4.5: "Productivity - Space",
    4.6: "Productivity - Downtime",
    4.8: "Productivity - Admin",
}

NAICS_SECTORS = {
    11: "Agriculture/Forestry/Fishing",
    21: "Mining/Oil & Gas",
    22: "Utilities",
    23: "Construction",
    31: "Food/Beverage/Tobacco",
    32: "Chemical/Plastics/Paper/Rubber",
    33: "Metal/Machinery/Electronics",
    42: "Wholesale Trade",
    44: "Retail Trade",
    48: "Transportation",
    51: "Information",
    52: "Finance/Insurance",
    53: "Real Estate",
    54: "Professional Services",
    56: "Administrative Services",
    61: "Educational Services",
    62: "Health Care",
    71: "Arts/Entertainment",
    72: "Accommodation/Food Services",
    81: "Other Services",
    92: "Public Administration",
}

STATE_ABBR = {
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
    "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
    "WV", "WY", "DC",
}

CENSUS_REGION_BY_STATE = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South", "VA": "South", "DC": "South", "WV": "South",
    "AL": "South", "KY": "South", "MS": "South", "TN": "South",
    "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West", "WA": "West",
}

STAGE1_CATEGORICAL_FEATURES = ["sector_name", "state", "census_region"]

STAGE2_BASELINE_FEATURES = [
    "log_employees",
    "log_plant_area",
    "log_annual_elec_kwh",
    "log_annual_gas_mmbtu",
    "fy_norm",
    "sector_name",
    "state",
    "arc",
    "arc_2digit",
]

STAGE2_INTENSITY_FEATURES = [
    "log_kwh_per_employee",
    "log_kwh_per_sqft",
    "log_mmbtu_per_employee",
    "log_total_energy_per_employee",
    "log_total_energy_per_sqft",
    "gas_fraction_total_energy",
    "log_total_energy",
]

STAGE2_ZSCORE_FEATURES = [
    "z_annual_elec_kwh",
    "z_annual_gas_mmbtu",
    "z_total_energy",
    "z_kwh_per_employee",
    "z_kwh_per_sqft",
    "z_total_energy_per_employee",
    "z_total_energy_per_sqft",
]

STAGE2_TARGET_ENCODING_FEATURE = "arc_target_encode"

STAGE2_CATEGORICAL_FEATURES = ["sector_name", "state", "arc", "arc_2digit"]

STAGE2_PAPER_FEATURES = (
    STAGE2_BASELINE_FEATURES
    + STAGE2_INTENSITY_FEATURES
    + STAGE2_ZSCORE_FEATURES
    + [STAGE2_TARGET_ENCODING_FEATURE]
)

STAGE2_EXTENDED_NUMERIC_FEATURES = [
    "plant_area_missing",
    "annual_electric_cost_missing",
    "annual_gas_usage_missing",
    "annual_gas_cost_missing",
    "elec_rate_per_kwh_filled",
    "gas_rate_per_mmbtu_filled",
    "state_year_electricity_price_mmbtu",
    "state_year_natural_gas_price_mmbtu",
    "state_year_elec_gas_ratio",
    "current_electricity_price_mmbtu",
    "current_natural_gas_price_mmbtu",
    "current_elec_gas_ratio",
    "electric_share_total_energy",
    "thermal_share_total_energy",
]

STAGE2_EXTENDED_CATEGORICAL_FEATURES = ["census_region", "primary_source_group", "primary_fuel_bucket"]

STAGE2_EXTENDED_FEATURES = (
    STAGE2_PAPER_FEATURES
    + STAGE2_EXTENDED_NUMERIC_FEATURES
    + STAGE2_EXTENDED_CATEGORICAL_FEATURES
)

STAGE2_FEATURE_SETS = {
    "paper": STAGE2_PAPER_FEATURES,
    "source_aware": STAGE2_EXTENDED_FEATURES,
}

STAGE2_CATEGORICAL_FEATURES_BY_SET = {
    "paper": STAGE2_CATEGORICAL_FEATURES,
    "source_aware": STAGE2_CATEGORICAL_FEATURES + STAGE2_EXTENDED_CATEGORICAL_FEATURES,
}


def load_emission_factors() -> dict:
    path = DATA_DIR / "emission_factors.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "natural_gas": 53.06,
        "petroleum": 74.14,
        "coal": 95.35,
        "electricity": 131.88,
        "weighted": 59.91,
    }

