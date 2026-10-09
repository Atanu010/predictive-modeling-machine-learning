# Project Report: Diabetes Progression Prediction

## Objective
Compare three regression algorithms for predicting the quantitative disease-progression measure in scikit-learn's built-in diabetes dataset.

## Data and Method
The experiment uses 442 observations and a fixed 80/20 train/test split (`random_state=42`). A median imputer is fitted within each model pipeline. The engineered `bmi_age_interaction` feature multiplies the BMI and age features. Linear Regression also uses standardization; the tree models do not require it.

## Holdout Performance
| Model | MAE | RMSE | R² |
|:--|--:|--:|--:|
| Linear Regression | 42.572 | 53.480 | 0.460 |
| Random Forest | 43.791 | 54.149 | 0.447 |
| Decision Tree | 44.519 | 55.617 | 0.416 |

The lowest holdout RMSE belongs to **Linear Regression**, which is saved as the selected model. MAE and RMSE are in target units; R² measures the fraction of holdout target variance explained.

## Visualizations
- `images/model_comparison.png`: MAE, RMSE, and R² comparison.
- `images/feature_importance.png`: Random Forest impurity-based feature importance.
- `images/predictions.png`: selected model predictions against holdout actuals.

## Interpretation and Limitations
These results are an educational benchmark, not a clinical model. The dataset is small and the fixed split is only one estimate of generalization. Impurity-based importance can favor features with many potential split points and does not establish causality. Further work should use cross-validation, tune models within each training fold, and validate on a suitable external population.
