# GHG Soil Moisture and Temperature ML Pipeline

This project contains a complete machine learning pipeline for greenhouse gas and agricultural soil measurement data. It cleans the raw CSV, engineers date and seasonal features, trains regression models, evaluates them with cross-validation and a time-based holdout split, and saves trained models plus model diagnostics.

## Project Highlights

- Cleans greenhouse gas measurement data with source metadata rows.
- Converts GHG measurement columns into readable feature names.
- Handles missing values with numeric and categorical preprocessing pipelines.
- Trains multiple regression models for:
  - soil moisture at 5 cm
  - soil temperature at 5 cm
- Uses a time-based holdout split using the latest year for final testing.
- Produces model leaderboards, data quality reports, predictions, feature importance, and actual-vs-predicted plots.

## Files

- `outputs/ghg_ml_pipeline.py` - main ML pipeline script
- `outputs/requirements.txt` - Python dependencies
- `outputs/COLAB_RUN_INSTRUCTIONS.md` - Google Colab run steps
- `outputs/ghg_ml_pipeline_outputs/ML_PIPELINE_REPORT.md` - generated model report
- `outputs/ghg_ml_pipeline_outputs/model_leaderboard.csv` - model comparison table
- `outputs/ghg_ml_pipeline_outputs/model_metrics.json` - full metrics and artifact paths
- `outputs/ghg_ml_pipeline_outputs/data_quality_summary.csv` - data quality summary
- `outputs/ghg_ml_pipeline_outputs/*_model.joblib` - trained model files
- `outputs/ghg_ml_pipeline_outputs/*_actual_vs_predicted.png` - diagnostic plots

## Setup

```bash
pip install -r outputs/requirements.txt
```

## Run

```bash
python outputs/ghg_ml_pipeline.py --input "path/to/ghg (1).csv"
```

Optional:

```bash
python outputs/ghg_ml_pipeline.py --input "path/to/ghg (1).csv" --output-dir outputs/ghg_ml_pipeline_outputs --cv-folds 3
```

## Verified Results

The pipeline was verified on `ghg (1).csv` with 18,762 usable rows after removing metadata rows.

- Soil moisture best model: Elastic Net, holdout R2 about 0.3023
- Soil temperature best model: Ridge Regression, holdout R2 about 0.5634

See `outputs/ghg_ml_pipeline_outputs/ML_PIPELINE_REPORT.md` for the detailed model leaderboard.
