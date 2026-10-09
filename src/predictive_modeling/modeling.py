"""End-to-end diabetes progression regression experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
TARGET_NAME = "disease_progression"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add an interaction between BMI and age to a feature DataFrame."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> FeatureEngineer:
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.feature_names_out_ = list(features.columns)
        if "bmi" in features.columns and "age" in features.columns:
            self.feature_names_out_.append("bmi_age_interaction")
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = features.copy()
        if "bmi" in transformed.columns and "age" in transformed.columns:
            transformed["bmi_age_interaction"] = (
                transformed["bmi"] * transformed["age"]
            )
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the built-in diabetes dataset without requiring network access."""
    dataset = load_diabetes(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.rename(TARGET_NAME)
    return features, target


def build_models() -> dict[str, Pipeline]:
    """Create comparable preprocessing and regression pipelines."""
    linear_model = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    decision_tree = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                DecisionTreeRegressor(
                    max_depth=4,
                    min_samples_leaf=8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "Linear Regression": linear_model,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }


def _evaluate(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(target, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(target, predictions))),
        "R2": float(r2_score(target, predictions)),
    }


def _save_visualizations(
    output_dir: Path,
    metrics: dict[str, dict[str, float]],
    fitted_models: dict[str, Pipeline],
    target_test: pd.Series,
    predictions: dict[str, np.ndarray],
) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x_positions = np.arange(len(model_names))
    bar_width = 0.36
    axes[0].bar(
        x_positions - bar_width / 2,
        [metrics[name]["MAE"] for name in model_names],
        bar_width,
        label="MAE",
        color="#2563eb",
    )
    axes[0].bar(
        x_positions + bar_width / 2,
        [metrics[name]["RMSE"] for name in model_names],
        bar_width,
        label="RMSE",
        color="#f97316",
    )
    axes[0].set_xticks(x_positions, model_names, rotation=12, ha="right")
    axes[0].set_ylabel("Error (lower is better)")
    axes[0].set_title("Holdout error")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        x_positions,
        [metrics[name]["R2"] for name in model_names],
        color="#16856b",
    )
    axes[1].set_xticks(x_positions, model_names, rotation=12, ha="right")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].set_title("Holdout explained variance")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(images_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    forest = fitted_models["Random Forest"].named_steps["regressor"]
    engineer = fitted_models["Random Forest"].named_steps["feature_engineering"]
    feature_names = engineer.get_feature_names_out()
    importance_order = np.argsort(forest.feature_importances_)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(
        feature_names[importance_order],
        forest.feature_importances_[importance_order],
        color="#16856b",
    )
    axis.set_xlabel("Random Forest impurity-based importance")
    axis.set_title("Feature importance")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(images_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    best_name = min(metrics, key=lambda name: metrics[name]["RMSE"])
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(target_test, predictions[best_name], alpha=0.72, color="#2563eb")
    limits = [
        min(float(target_test.min()), float(predictions[best_name].min())),
        max(float(target_test.max()), float(predictions[best_name].max())),
    ]
    axis.plot(
        limits,
        limits,
        linestyle="--",
        color="#f97316",
        label="Perfect prediction",
    )
    axis.set_xlabel("Actual progression")
    axis.set_ylabel("Predicted progression")
    axis.set_title(f"Actual vs predicted: {best_name}")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(images_dir / "predictions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    output_dir: Path,
    metrics: dict[str, dict[str, float]],
    best_name: str,
    sample_count: int,
) -> None:
    ranking = sorted(metrics, key=lambda name: metrics[name]["RMSE"])
    rows = [
        "| Model | MAE | RMSE | R² |",
        "|:--|--:|--:|--:|",
    ]
    rows.extend(
        f"| {name} | {metrics[name]['MAE']:.3f} | "
        f"{metrics[name]['RMSE']:.3f} | {metrics[name]['R2']:.3f} |"
        for name in ranking
    )
    report = "\n".join(
        [
            "# Project Report: Diabetes Progression Prediction",
            "",
            "## Objective",
            "Compare three regression algorithms for predicting the quantitative "
            "disease-progression measure in scikit-learn's built-in diabetes dataset.",
            "",
            "## Data and Method",
            f"The experiment uses {sample_count} observations and a fixed 80/20 "
            "train/test split (`random_state=42`). A median imputer is fitted "
            "within each model pipeline. The engineered `bmi_age_interaction` "
            "feature multiplies the BMI and age features. Linear Regression also "
            "uses standardization; the tree models do not require it.",
            "",
            "## Holdout Performance",
            *rows,
            "",
            f"The lowest holdout RMSE belongs to **{best_name}**, which is saved "
            "as the selected model. MAE and RMSE are in target units; R² measures "
            "the fraction of holdout target variance explained.",
            "",
            "## Visualizations",
            "- `images/model_comparison.png`: MAE, RMSE, and R² comparison.",
            "- `images/feature_importance.png`: Random Forest impurity-based "
            "feature importance.",
            "- `images/predictions.png`: selected model predictions against "
            "holdout actuals.",
            "",
            "## Interpretation and Limitations",
            "These results are an educational benchmark, not a clinical model. "
            "The dataset is small and the fixed split is only one estimate of "
            "generalization. Impurity-based importance can favor features with "
            "many potential split points and does not establish causality. "
            "Further work should use cross-validation, tune models within each "
            "training fold, and validate on a suitable external population.",
            "",
        ]
    )
    (output_dir / "project_report.md").write_text(report, encoding="utf-8")


def run_experiment(output_dir: str | Path = ".") -> dict[str, Any]:
    """Fit, evaluate, visualize, and save the best model and experiment report."""
    output_path = Path(output_dir)
    features, target = load_dataset()
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    fitted_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in build_models().items():
        model.fit(features_train, target_train)
        model_predictions = model.predict(features_test)
        fitted_models[name] = model
        predictions[name] = model_predictions
        metrics[name] = _evaluate(target_test, model_predictions)

    best_name = min(metrics, key=lambda name: metrics[name]["RMSE"])
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": fitted_models[best_name],
        "model_name": best_name,
        "feature_columns": list(features.columns),
        "target_name": TARGET_NAME,
        "metrics": metrics[best_name],
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, models_dir / "best_model.pkl")
    _save_visualizations(output_path, metrics, fitted_models, target_test, predictions)
    _write_report(output_path, metrics, best_name, len(features))

    return {
        "metrics": metrics,
        "best_model": best_name,
        "artifact_path": models_dir / "best_model.pkl",
        "images_dir": output_path / "images",
        "report_path": output_path / "project_report.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory in which to write images/, models/, and project_report.md.",
    )
    arguments = parser.parse_args()
    results = run_experiment(arguments.output_dir)
    print("Holdout metrics:")
    formatted_metrics = pd.DataFrame(results["metrics"]).T.to_string(
        float_format=lambda value: f"{value:.3f}"
    )
    print(formatted_metrics)
    print(f"\nSelected model: {results['best_model']}")
    print(f"Model: {results['artifact_path']}")
    print(f"Charts: {results['images_dir']}")
    print(f"Report: {results['report_path']}")


if __name__ == "__main__":
    main()