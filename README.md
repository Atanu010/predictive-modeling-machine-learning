# Predictive Modeling: Diabetes Progression

A reproducible regression project comparing Linear Regression, Decision Tree,
and Random Forest models on scikit-learn's diabetes dataset. The workflow is
offline after dependencies are installed and produces evaluation metrics,
publication-ready charts, a serialized best model, and a project report.

## Project Highlights

- Median imputation and standardization for the linear model, with tree models
	using median imputation without scaling.
- BMI-by-age interaction feature engineering.
- Fixed train/test split and consistent MAE, RMSE, and R² evaluation.
- Automatically generated model-comparison, prediction, and feature-importance
	charts.
- Automated tests for preprocessing, model metrics, and artifact generation.

## Quick Start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m predictive_modeling
pytest
```

The scikit-learn diabetes dataset is fetched from the installed package; no
separate data download is required. Running the experiment creates
`images/model_comparison.png`, `images/feature_importance.png`,
`images/predictions.png`, `models/best_model.pkl`, and refreshes
`project_report.md`.

To run the walkthrough, launch Jupyter and open
[`notebooks/predictive_modeling_walkthrough.ipynb`](notebooks/predictive_modeling_walkthrough.ipynb):

```bash
jupyter notebook
```

Run all notebook cells from top to bottom. The notebook invokes the same
experiment code as the command-line workflow, so its charts and artifacts are
generated automatically.

## Results

The holdout metrics for all three models and the selected model are recorded in
[`project_report.md`](project_report.md) after running the experiment. The
Random Forest feature-importance chart is a model-based view of predictive
signal, not a causal explanation.

## Repository Structure

```text
.
├── images/                     # Generated charts
├── models/                     # Serialized selected model
├── notebooks/                  # End-to-end walkthrough
├── src/predictive_modeling/    # Reusable training and evaluation code
├── tests/                      # Automated tests
├── project_report.md           # Experiment results and interpretation
└── pyproject.toml              # Dependencies and tooling configuration
```

## Reproducibility and Limitations

The holdout set and Random Forest use fixed random seeds. The dataset is a
small, built-in teaching dataset; results should not be treated as clinical
advice or evidence of performance on a production population. The experiment
does not tune hyperparameters or claim causal relationships.