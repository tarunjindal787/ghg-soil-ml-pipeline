# Colab Run Instructions for `ghg_ml_pipeline.py`

Use these steps inside your Colab notebook:

1. Upload `ghg (1).csv` to the Colab session files panel.
2. Upload `ghg_ml_pipeline.py` from this outputs folder, or paste its code into a Colab cell.
3. Optional: install/verify dependencies:

```python
!pip install -q pandas numpy scikit-learn joblib matplotlib
```

4. Run:

```python
!python ghg_ml_pipeline.py --input "/content/ghg (1).csv" --output-dir "/content/ghg_ml_pipeline_outputs"
```

5. View results:

```python
import json
from pathlib import Path

metrics_path = Path("/content/ghg_ml_pipeline_outputs/model_metrics.json")
print(metrics_path.read_text())
```

6. Download generated files:

```python
from google.colab import files

for name in [
    "ML_PIPELINE_REPORT.md",
    "model_metrics.json",
    "model_leaderboard.csv",
    "data_quality_summary.csv",
    "ghg_cleaned_modeling_data.csv",
    "soil_moisture_model.joblib",
    "soil_temperature_model.joblib",
    "soil_moisture_5cm_holdout_predictions.csv",
    "soil_temperature_5cm_holdout_predictions.csv",
    "soil_moisture_5cm_feature_importance.csv",
    "soil_temperature_5cm_feature_importance.csv",
    "soil_moisture_5cm_actual_vs_predicted.png",
    "soil_temperature_5cm_actual_vs_predicted.png",
]:
    files.download(f"/content/ghg_ml_pipeline_outputs/{name}")
```

If you want Codex to edit the notebook directly, set the Colab sharing permission to "Anyone with the link can view" or download the notebook as `.ipynb` and attach it here.
