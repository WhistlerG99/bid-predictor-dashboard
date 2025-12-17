# Bid Predictor Dashboard

Interactive Dash application for exploring bid acceptance probabilities from MLflow-logged models and parquet snapshot datasets.

## Prerequisites
- Python 3.9+
- Access to a parquet dataset of bid snapshots (path provided in the UI or via `DEFAULT_DATASET_PATH`).
- Access to an MLflow tracking server and registered model (URI provided in the UI or via `MLFLOW_AWS_ARN`).
- Optional: a `.env` file in the repository root to store environment variables.

## Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade `pip` and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration
Set the following environment variables directly or in a `.env` file before starting the app:
- `MLFLOW_AWS_ARN`: MLflow tracking URI used by the dashboard to load registered models.
- `DEFAULT_DATASET_PATH` (optional): default parquet path pre-filled in the UI.
- `PORT` (optional): port for the Dash server (defaults to `8000`).

## Running the Dash App
From the repository root (with the virtual environment activated), start the dashboard:
```bash
python dash_app.py
```
Then open http://localhost:8000 in your browser. You can adjust the dataset path, MLflow tracking URI, and model details within the UI.

## Testing
Run the test suite from the repository root:
```bash
PYTHONPATH=. pytest -q
```
