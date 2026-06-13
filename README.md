# GHG Soil Moisture and Temperature Prediction ML Pipeline

## Project Overview

This repository contains a complete machine learning pipeline for agricultural greenhouse gas and soil measurement data. The goal of the project is to clean raw environmental data, build reusable preprocessing logic, train regression models, and predict two important soil properties:

- Soil moisture at 5 cm depth
- Soil temperature at 5 cm depth

The pipeline is designed for greenhouse gas, soil, and sustainable agriculture datasets where measurements may contain missing values, metadata rows, categorical experimental conditions, and time-dependent patterns.

## Problem Statement

Agricultural field datasets often contain greenhouse gas flux measurements, soil chemistry readings, sampling methods, chamber positions, and date-based observations. These datasets need careful preprocessing before they can be used for machine learning.

This project builds an end-to-end regression pipeline that:

- Extracts and cleans raw GHG CSV data.
- Removes non-observation metadata rows.
- Converts coded GHG columns into readable feature names.
- Handles missing numerical and categorical values.
- Engineers time and seasonal features.
- Trains multiple regression models.
- Evaluates models using cross-validation and a final time-based holdout test.
- Saves model outputs, predictions, reports, plots, and trained model files.

## Dataset Summary

The pipeline was tested on `ghg (1).csv`.

- Raw data type: CSV
- Usable rows after metadata removal: 18,762
- Time range: 2011 to 2015
- Target variables:
  - `soil_moisture_5cm`
  - `soil_temperature_5cm`
- Input feature groups:
  - Plot and sampling identifiers
  - Measurement year and date-derived features
  - GHG flux measurements
  - Model quality/R2 measurement columns
  - Sampling method
  - Chamber position
  - Soil nitrogen features where available

## Machine Learning Pipeline

The main pipeline is implemented in:

`outputs/ghg_ml_pipeline.py`

### 1. Data Ingestion

The script reads the raw CSV file and removes source metadata rows such as `description` and `units`, which are not real observations.

### 2. Data Cleaning

The cleaning stage:

- Converts blank strings and invalid numeric values to missing values.
- Converts GHG measurement columns to numeric format.
- Parses the `date` column.
- Standardizes text columns such as method and chamber position.
- Drops columns that are almost completely empty.

### 3. Feature Engineering

The pipeline creates additional date-based features:

- `month`
- `day_of_year`
- `quarter`
- `season`
- `is_growing_season`

These features help the models capture seasonal agricultural patterns.

### 4. Leakage-Safe Target Handling

The pipeline predicts soil moisture and soil temperature separately. When training one target, both target columns are removed from the feature set. This prevents the model from accidentally learning from another directly measured soil label.

### 5. Model Training

The following regression models are compared:

- Ridge Regression
- Elastic Net
- Random Forest Regressor
- Extra Trees Regressor
- Histogram Gradient Boosting Regressor

### 6. Model Evaluation

The project uses:

- 3-fold cross-validation on the training data
- A time-based holdout split using the latest year as the final test set
- MAE, RMSE, and R2 metrics

Using a time-based holdout gives a more realistic evaluation for environmental datasets because it tests whether models generalize to future observations.

### 7. Output Generation

The pipeline saves:

- Cleaned modeling dataset
- Data quality summary
- Model leaderboard
- Full metrics JSON
- Holdout predictions
- Feature importance files
- Actual-vs-predicted plots
- Trained `.joblib` models
- Markdown model report

## Project Structure

```text
.
├── README.md
├── .gitignore
└── outputs
    ├── ghg_ml_pipeline.py
    ├── requirements.txt
    ├── COLAB_RUN_INSTRUCTIONS.md
    └── ghg_ml_pipeline_outputs
        ├── ML_PIPELINE_REPORT.md
        ├── data_quality_summary.csv
        ├── ghg_cleaned_modeling_data.csv
        ├── model_leaderboard.csv
        ├── model_metrics.json
        ├── soil_moisture_5cm_actual_vs_predicted.png
        ├── soil_moisture_5cm_feature_importance.csv
        ├── soil_moisture_5cm_holdout_predictions.csv
        ├── soil_moisture_model.joblib
        ├── soil_temperature_5cm_actual_vs_predicted.png
        ├── soil_temperature_5cm_feature_importance.csv
        ├── soil_temperature_5cm_holdout_predictions.csv
        └── soil_temperature_model.joblib
```

## How to Run

Install dependencies:

```bash
pip install -r outputs/requirements.txt
```

Run the pipeline:

```bash
python outputs/ghg_ml_pipeline.py --input "path/to/ghg (1).csv"
```

Run with a custom output directory and cross-validation setting:

```bash
python outputs/ghg_ml_pipeline.py --input "path/to/ghg (1).csv" --output-dir outputs/ghg_ml_pipeline_outputs --cv-folds 3
```

## Google Colab

Colab instructions are included in:

`outputs/COLAB_RUN_INSTRUCTIONS.md`

Basic Colab command:

```python
!python ghg_ml_pipeline.py --input "/content/ghg (1).csv" --output-dir "/content/ghg_ml_pipeline_outputs"
```

## Model Results

### Soil Moisture Prediction

Target: `soil_moisture_5cm`

- Split strategy: latest year holdout, `2015`
- Rows used: 13,208
- Training rows: 10,998
- Test rows: 2,210
- Best model: Elastic Net

| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |
|---|---:|---:|---:|---:|
| Elastic Net | 9.0217 | 0.3533 | 9.0511 | 0.3023 |
| Ridge | 8.8944 | 0.3715 | 9.2394 | 0.2729 |
| Histogram Gradient Boosting | 5.2301 | 0.7827 | 9.5547 | 0.2225 |
| Extra Trees | 4.8535 | 0.8128 | 10.1689 | 0.1193 |
| Random Forest | 4.9811 | 0.8029 | 10.2939 | 0.0975 |

### Soil Temperature Prediction

Target: `soil_temperature_5cm`

- Split strategy: latest year holdout, `2015`
- Rows used: 14,913
- Training rows: 12,711
- Test rows: 2,202
- Best model: Ridge Regression

| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |
|---|---:|---:|---:|---:|
| Ridge | 4.3910 | 0.6145 | 4.3220 | 0.5634 |
| Elastic Net | 4.4416 | 0.6055 | 4.3615 | 0.5554 |
| Extra Trees | 1.5875 | 0.9495 | 4.3623 | 0.5552 |
| Histogram Gradient Boosting | 1.9197 | 0.9263 | 4.4458 | 0.5380 |
| Random Forest | 1.7471 | 0.9390 | 4.9533 | 0.4265 |

## Output Files

| File | Description |
|---|---|
| `ML_PIPELINE_REPORT.md` | Human-readable model report |
| `model_metrics.json` | Full structured metrics and artifact paths |
| `model_leaderboard.csv` | Model comparison table |
| `data_quality_summary.csv` | Missing value and column summary |
| `ghg_cleaned_modeling_data.csv` | Cleaned dataset used for modeling |
| `*_holdout_predictions.csv` | Actual, predicted, and residual values |
| `*_feature_importance.csv` | Feature importance or permutation importance |
| `*_actual_vs_predicted.png` | Diagnostic plots |
| `*_model.joblib` | Saved trained models |

## Pipeline Advantages

- End-to-end workflow from raw CSV to trained models.
- Handles real-world messy environmental data.
- Uses reusable Scikit-learn pipelines for preprocessing and modeling.
- Prevents target leakage by removing target columns from features.
- Uses time-based holdout testing for realistic future-year evaluation.
- Compares multiple regression algorithms.
- Saves outputs needed for review, reporting, and future prediction.
- Includes Colab instructions for easy notebook execution.

## Pipeline Limitations

- The soil moisture model has moderate predictive power, with holdout R2 around 0.30.
- Some GHG and soil chemistry columns have high missingness, which limits model performance.
- The dataset used here has 18,762 usable rows, not over 100,000 rows.
- No external weather, irrigation, crop, or treatment-management variables are included.
- Hyperparameter tuning is limited to practical default settings.
- Tree-based models show strong cross-validation scores but weaker future-year holdout performance, suggesting possible temporal shift or overfitting.

## Future Improvements

- Add weather data such as rainfall, humidity, solar radiation, and air temperature.
- Add crop type, fertilizer treatment, irrigation, and field management features.
- Use grouped validation by plot or experimental site.
- Tune models with randomized search or Bayesian optimization.
- Add SHAP-based explainability for richer feature interpretation.
- Build a Streamlit dashboard for interactive predictions and visualizations.
- Store large datasets and model binaries with Git LFS for larger production projects.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Conclusion

This project demonstrates a complete machine learning workflow for agricultural environmental data. It converts raw greenhouse gas and soil measurements into a reproducible modeling pipeline, evaluates multiple regression approaches, and produces interpretable outputs for soil moisture and soil temperature prediction.
