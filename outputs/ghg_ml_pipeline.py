"""
Production-style ML pipeline for greenhouse gas and soil measurement data.

What this pipeline does:
1. Ingests the raw CSV and removes source metadata rows.
2. Cleans numeric blanks, parses dates, and engineers seasonal features.
3. Builds leakage-aware feature sets for soil moisture and soil temperature.
4. Compares multiple regression models using cross-validation and a final holdout set.
5. Saves trained models, metrics, predictions, feature importance, and a readable report.

Example:
    python outputs/ghg_ml_pipeline.py --input "C:\\Users\\jinda\\Downloads\\ghg (1).csv"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42

COLUMN_DESCRIPTIONS = {
    "GHG01": "co2_c_flux_kg_ha_day",
    "GHG02": "co2_r2",
    "GHG03": "n2o_n_flux_g_ha_day",
    "GHG04": "n2o_r2",
    "GHG05": "nh3_n_flux_g_ha_day",
    "GHG06": "nh3_r2",
    "GHG07": "ch4_c_flux_g_ha_day",
    "GHG08": "ch4_r2",
    "GHG11": "soil_moisture_5cm",
    "GHG12": "soil_temperature_5cm",
    "GHG13": "soil_nitrate_n_0_10cm",
    "GHG14": "soil_ammonium_n_0_10cm",
}

TARGETS = {
    "soil_moisture_5cm": "soil_moisture_model.joblib",
    "soil_temperature_5cm": "soil_temperature_model.joblib",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_ghg_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = pd.read_csv(
        path,
        na_values=["", " ", "NA", "N/A", "NaN", "nan", "null", "-"],
        keep_default_na=True,
    )
    raw = raw[~raw["uniqueid"].astype(str).str.lower().isin({"description", "units"})]
    raw = raw.rename(columns=COLUMN_DESCRIPTIONS)
    raw.columns = [str(col).strip() for col in raw.columns]
    return raw.reset_index(drop=True)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = ["plotid", "year", "subsample", *COLUMN_DESCRIPTIONS.values()]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["is_growing_season"] = df["month"].between(4, 10).astype("Int64")
    df["season"] = np.select(
        [
            df["month"].isin([12, 1, 2]),
            df["month"].isin([3, 4, 5]),
            df["month"].isin([6, 7, 8]),
            df["month"].isin([9, 10, 11]),
        ],
        ["winter", "spring", "summer", "fall"],
        default=pd.NA,
    )

    for column in ["uniqueid", "method", "position"]:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip().replace("", pd.NA)

    missing_rate = df.isna().mean()
    sparse_columns = missing_rate[missing_rate > 0.98].index.tolist()
    sparse_columns = [col for col in sparse_columns if col not in TARGETS]
    return df.drop(columns=sparse_columns)


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_pct": [round(float(df[col].isna().mean() * 100), 2) for col in df.columns],
            "unique_count": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )
    return summary.sort_values(["missing_pct", "column"], ascending=[False, True])


def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    modeling_df = df.dropna(subset=[target]).copy()

    # Exclude both target variables to avoid leakage between measured soil labels.
    leakage_columns = [*TARGETS.keys(), "date"]
    feature_columns = [col for col in modeling_df.columns if col not in leakage_columns]
    X = modeling_df[feature_columns]
    y = modeling_df[target].astype(float)

    if modeling_df["year"].nunique(dropna=True) > 1:
        test_year = int(modeling_df["year"].max())
        train_mask = modeling_df["year"] < test_year
        if train_mask.sum() >= 100 and (~train_mask).sum() >= 50:
            return X[train_mask], X[~train_mask], y[train_mask], y[~train_mask], f"time_holdout_year_{test_year}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, "random_80_20_holdout"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool", "Int64"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def candidate_models() -> Dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.15, max_iter=5000, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=90,
            max_depth=14,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=90,
            max_depth=14,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=180,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model_cv(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int
) -> Dict[str, Optional[float]]:
    if cv_folds < 2 or len(y_train) < cv_folds * 20:
        return {"cv_mae": None, "cv_rmse": None, "cv_r2": None}

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
        error_score="raise",
    )
    return {
        "cv_mae": float(-scores["test_mae"].mean()),
        "cv_rmse": float(-scores["test_rmse"].mean()),
        "cv_r2": float(scores["test_r2"].mean()),
    }


def get_feature_names(pipeline: Pipeline, X: pd.DataFrame) -> List[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return list(X.columns)


def save_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
    max_rows: int = 1200,
) -> Optional[Path]:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline, X_test)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    else:
        sample_X = X_test.sample(min(len(X_test), max_rows), random_state=RANDOM_STATE)
        sample_y = y_test.loc[sample_X.index]
        result = permutation_importance(
            pipeline,
            sample_X,
            sample_y,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )

    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)
    return output_path


def save_prediction_plot(
    y_true: pd.Series, y_pred: np.ndarray, target: str, output_path: Path
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    ax.scatter(y_true, y_pred, alpha=0.45, s=18, edgecolors="none")
    low = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    high = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    ax.plot([low, high], [low, high], color="#C43C35", linewidth=2)
    ax.set_title(f"Actual vs Predicted: {target}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def train_for_target(df: pd.DataFrame, target: str, output_dir: Path, cv_folds: int) -> Dict[str, object]:
    X_train, X_test, y_train, y_test, split_strategy = split_features_target(df, target)

    leaderboard: List[Dict[str, object]] = []
    trained_models: Dict[str, Pipeline] = {}

    for model_name, model in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", clone(model)),
            ]
        )
        cv_metrics = evaluate_model_cv(pipeline, X_train, y_train, cv_folds)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        holdout_metrics = regression_metrics(y_test, predictions)
        leaderboard.append({"model": model_name, **cv_metrics, **holdout_metrics})
        trained_models[model_name] = pipeline

    ranked = sorted(leaderboard, key=lambda item: (item["r2"], -item["rmse"]), reverse=True)
    best_name = ranked[0]["model"]
    best_pipeline = trained_models[best_name]
    best_predictions = best_pipeline.predict(X_test)

    model_path = output_dir / TARGETS[target]
    predictions_path = output_dir / f"{target}_holdout_predictions.csv"
    importance_path = output_dir / f"{target}_feature_importance.csv"
    plot_path = output_dir / f"{target}_actual_vs_predicted.png"

    joblib.dump(best_pipeline, model_path)
    pd.DataFrame(
        {
            "actual": y_test,
            "predicted": best_predictions,
            "residual": y_test - best_predictions,
        }
    ).to_csv(predictions_path, index=False)
    save_feature_importance(best_pipeline, X_test, y_test, importance_path)
    saved_plot = save_prediction_plot(y_test, best_predictions, target, plot_path)

    return {
        "target": target,
        "split_strategy": split_strategy,
        "rows_used": int(len(y_train) + len(y_test)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": list(X_train.columns),
        "best_model": best_name,
        "saved_model": str(model_path),
        "saved_predictions": str(predictions_path),
        "saved_feature_importance": str(importance_path),
        "saved_plot": str(saved_plot) if saved_plot else None,
        "leaderboard": ranked,
    }


def write_report(report: Dict[str, object], output_dir: Path) -> Path:
    lines = [
        "# GHG Soil ML Pipeline Report",
        "",
        f"Input file: `{report['input_file']}`",
        f"Cleaned rows: **{report['total_rows_after_metadata_removal']}**",
        f"Cleaned dataset: `{report['cleaned_dataset']}`",
        "",
        "## Model Results",
        "",
    ]

    for target_report in report["targets"]:
        best = target_report["leaderboard"][0]
        lines.extend(
            [
                f"### {target_report['target']}",
                "",
                f"- Split: `{target_report['split_strategy']}`",
                f"- Rows used: {target_report['rows_used']} ({target_report['train_rows']} train, {target_report['test_rows']} test)",
                f"- Best model: **{target_report['best_model']}**",
                f"- Holdout MAE: **{best['mae']:.4f}**",
                f"- Holdout RMSE: **{best['rmse']:.4f}**",
                f"- Holdout R2: **{best['r2']:.4f}**",
                "",
                "| Model | CV RMSE | CV R2 | Holdout RMSE | Holdout R2 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in target_report["leaderboard"]:
            cv_rmse = "" if row["cv_rmse"] is None else f"{row['cv_rmse']:.4f}"
            cv_r2 = "" if row["cv_r2"] is None else f"{row['cv_r2']:.4f}"
            lines.append(
                f"| {row['model']} | {cv_rmse} | {cv_r2} | {row['rmse']:.4f} | {row['r2']:.4f} |"
            )
        lines.append("")

    report_path = output_dir / "ML_PIPELINE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_pipeline(input_path: Path, output_dir: Path, cv_folds: int) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = clean_and_engineer(load_ghg_csv(input_path))

    clean_path = output_dir / "ghg_cleaned_modeling_data.csv"
    quality_path = output_dir / "data_quality_summary.csv"
    leaderboard_path = output_dir / "model_leaderboard.csv"

    df.to_csv(clean_path, index=False)
    data_quality_summary(df).to_csv(quality_path, index=False)

    target_summaries = [train_for_target(df, target, output_dir, cv_folds) for target in TARGETS]

    leaderboard_rows = []
    for target_summary in target_summaries:
        for row in target_summary["leaderboard"]:
            leaderboard_rows.append({"target": target_summary["target"], **row})
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)

    report = {
        "input_file": str(input_path),
        "cleaned_dataset": str(clean_path),
        "data_quality_summary": str(quality_path),
        "model_leaderboard": str(leaderboard_path),
        "total_rows_after_metadata_removal": int(len(df)),
        "columns": list(df.columns),
        "targets": target_summaries,
    }

    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["markdown_report"] = str(write_report(report, output_dir))
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    default_input = Path.home() / "Downloads" / "ghg (1).csv"
    default_output = Path(__file__).resolve().parent / "ghg_ml_pipeline_outputs"

    parser = argparse.ArgumentParser(description="Train and evaluate GHG soil regression models.")
    parser.add_argument("--input", type=Path, default=default_input, help="Input CSV path.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output directory.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds; set 0 to disable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_pipeline(args.input, args.output_dir, args.cv_folds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
