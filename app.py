from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from assessment_engine import run_assessment, SECTOR_OPTIONS, STATE_OPTIONS, SYSTEM_OPTIONS

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
STATIC = BASE / "static"

app = FastAPI(
    title="IAC Facility Assessment Tool",
    description="Pre-audit industrial energy screening using the standalone Paper 2 assessment models",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC / "index.html"))


class FacilityInput(BaseModel):
    sector: str = Field(..., description="Industry sector name")
    state: str = Field(..., description="2-letter US state code")
    employees: Optional[int] = Field(None, ge=1, description="Number of employees")
    plant_area_sqft: Optional[int] = Field(None, ge=100, description="Plant floor area in square feet")
    annual_elec_kwh: Optional[float] = Field(None, ge=0, description="Annual electricity usage (kWh)")
    annual_gas_mmbtu: Optional[float] = Field(None, ge=0, description="Annual natural gas usage (MMBtu)")
    annual_elec_cost: Optional[float] = Field(None, ge=0, description="Annual electricity cost (USD)")
    annual_gas_cost: Optional[float] = Field(None, ge=0, description="Annual natural gas cost (USD)")
    systems_present: Optional[List[str]] = Field(default_factory=list, description="Systems present or of concern at the facility")
    top_n_categories: Optional[int] = Field(6, ge=1, le=24)
    top_n_per_cat: Optional[int] = Field(3, ge=1, le=10)


@app.get("/api/options", summary="Get valid sector, state, and system options")
async def get_options():
    return {
        "sectors": SECTOR_OPTIONS,
        "states": STATE_OPTIONS,
        "systems": SYSTEM_OPTIONS,
    }


@app.post("/api/assess", summary="Run the facility assessment")
async def assess(facility: FacilityInput):
    try:
        return run_assessment(
            facility=facility.model_dump(exclude={"top_n_categories", "top_n_per_cat"}),
            top_n_categories=facility.top_n_categories,
            top_n_per_cat=facility.top_n_per_cat,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/health", summary="Health check")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )

