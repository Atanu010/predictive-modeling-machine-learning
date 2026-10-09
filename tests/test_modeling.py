"""Focused tests for preprocessing, evaluation, and generated artifacts."""

import joblib
import numpy as np
import pandas as pd
import pytest

from predictive_modeling.modeling import FeatureEngineer, build_models, run_experiment


def test_feature_engineering_adds_interaction_and_keeps_missing_values() -> None:
    features = pd.DataFrame(
        {"age": [0.1, np.nan, 0.2, 0.3], "bmi": [0.2, 0.3, 0.4, 0.5]}
    )

    engineered = FeatureEngineer().fit_transform(features)

    assert "bmi_age_interaction" in engineered.columns
    assert engineered.loc[0, "bmi_age_interaction"] == pytest.approx(0.02)
    assert np.isnan(engineered.loc[1, "bmi_age_interaction"])

    pipeline = build_models()["Linear Regression"]
    pipeline.fit(features, pd.Series([10.0, 20.0, 15.0, 30.0]))
    assert np.isfinite(pipeline.predict(features)).all()


def test_experiment_generates_metrics_plots_report_and_loadable_model(tmp_path) -> None:
    results = run_experiment(tmp_path)

    assert set(results["metrics"]) == {
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
    }
    for model_metrics in results["metrics"].values():
        assert set(model_metrics) == {"MAE", "RMSE", "R2"}
        assert all(np.isfinite(value) for value in model_metrics.values())
        assert model_metrics["MAE"] >= 0
        assert model_metrics["RMSE"] >= 0
    assert (tmp_path / "images" / "model_comparison.png").is_file()
    assert (tmp_path / "images" / "feature_importance.png").is_file()
    assert (tmp_path / "images" / "predictions.png").is_file()
    assert (tmp_path / "project_report.md").is_file()

    artifact = joblib.load(tmp_path / "models" / "best_model.pkl")
    assert artifact["model_name"] == results["best_model"]
    inference_row = pd.DataFrame(
        {column: [0.0] for column in artifact["feature_columns"]}
    )
    assert artifact["model"].predict(inference_row).shape == (1,)