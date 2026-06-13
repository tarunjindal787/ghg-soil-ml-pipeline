# GHG Soil ML Pipeline Report

Input file: `C:\Users\jinda\Downloads\ghg (1).csv`
Cleaned rows: **18762**
Cleaned dataset: `C:\Users\jinda\Documents\Codex\2026-06-13\hi\outputs\ghg_ml_pipeline_outputs\ghg_cleaned_modeling_data.csv`

## Model Results

### soil_moisture_5cm

- Split: `time_holdout_year_2015`
- Rows used: 13208 (10998 train, 2210 test)
- Best model: **elastic_net**
- Holdout MAE: **6.4658**
- Holdout RMSE: **9.0511**
- Holdout R2: **0.3023**

| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |
|---|---:|---:|---:|---:|
| elastic_net | 9.0217 | 0.3533 | 9.0511 | 0.3023 |
| ridge | 8.8944 | 0.3715 | 9.2394 | 0.2729 |
| hist_gradient_boosting | 5.2301 | 0.7827 | 9.5547 | 0.2225 |
| extra_trees | 4.8535 | 0.8128 | 10.1689 | 0.1193 |
| random_forest | 4.9811 | 0.8029 | 10.2939 | 0.0975 |

### soil_temperature_5cm

- Split: `time_holdout_year_2015`
- Rows used: 14913 (12711 train, 2202 test)
- Best model: **ridge**
- Holdout MAE: **3.4243**
- Holdout RMSE: **4.3220**
- Holdout R2: **0.5634**

| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |
|---|---:|---:|---:|---:|
| ridge | 4.3910 | 0.6145 | 4.3220 | 0.5634 |
| elastic_net | 4.4416 | 0.6055 | 4.3615 | 0.5554 |
| extra_trees | 1.5875 | 0.9495 | 4.3623 | 0.5552 |
| hist_gradient_boosting | 1.9197 | 0.9263 | 4.4458 | 0.5380 |
| random_forest | 1.7471 | 0.9390 | 4.9533 | 0.4265 |
