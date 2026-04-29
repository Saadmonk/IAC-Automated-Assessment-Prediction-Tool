# IAC Facility Assessment Tool

This repository contains a standalone local version of the current industrial facility assessment tool built from the Paper 2 prediction models.

It is meant for local use. A user can download the repository, install the required Python packages, and run the tool on their own machine.

## What the tool does

The tool accepts basic facility information such as:

- industry sector
- state
- number of employees
- plant area
- annual electricity use and cost
- annual natural gas use and cost
- systems present at the facility

It then returns:

- likely ARC recommendation categories
- a ranked recommendation list
- estimated technical energy savings
- estimated annual savings value
- portfolio totals for the facility

## Main improvements in this version

This version uses the newer assessment models instead of the earlier legacy tool logic.

It also adds two protections that make the tool more realistic:

- facility-scale guardrails, so predicted savings stay reasonable relative to facility size and annual utility spend
- system applicability filtering, so recommendations that depend on systems not present at the facility are moved out of the primary list

## What is included in this repository

This repository includes everything needed to run the tool locally:

- the FastAPI application
- the browser-based interface
- the trained model files
- the ARC lookup data
- the price-context data used at runtime
- the minimal runtime support code required to build model features

This repository does not include the full research workspace. It is a deployment-style local bundle for using the tool, not for retraining the models.

## Repository structure

```text
IAC_Facility_Assessment_Tool/
|-- app.py
|-- assessment_engine.py
|-- requirements.txt
|-- README.md
|-- start.bat
|-- start.sh
|-- .gitignore
|-- models/
|   |-- arc_category_models.joblib
|   |-- facility_savings_models.joblib
|   |-- arc_recommendation_statistics.csv
|   |-- facility_guardrails.csv
|   |-- facility_guardrails_meta.json
|   |-- arc_category_model_manifest.json
|   `-- savings_model_manifest.json
|-- data/
|   |-- arc_codes.csv
|   |-- emission_factors.json
|   `-- external/
|       `-- state_year_energy_prices.csv
|-- paper2/
|   |-- __init__.py
|   |-- common.py
|   |-- config.py
|   `-- features.py
`-- static/
    `-- index.html
```

## Requirements

- Python 3.10 or newer is recommended
- Windows, macOS, or Linux

## Run locally on Windows

Open PowerShell in the repository folder and run:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

- `http://localhost:8000`

Do not open `http://0.0.0.0:8000` in the browser. That is only the server bind address.

## Run locally on macOS or Linux

Open Terminal in the repository folder and run:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

- `http://localhost:8000`

## One-click start scripts

You can also use:

- `start.bat` on Windows
- `start.sh` on macOS or Linux

These scripts create a virtual environment if needed, install the required packages, and start the server.

## API endpoints

The local server provides these endpoints:

- `GET /` for the web interface
- `GET /api/options` for sector, state, and system options
- `POST /api/assess` to run the assessment
- `GET /api/health` for a health check
- `GET /docs` for the built-in FastAPI API docs

## Notes for GitHub users

GitHub is a good way to share this tool so others can download it and run it locally.

GitHub alone does not run the full tool as a live website. To use the tool from GitHub, a person should:

1. download or clone the repository
2. follow the setup steps in this README
3. run the app locally

## Institution

This tool bundle is labeled for:

- University of Louisiana at Lafayette
- Industrial Assessment Centers Program

