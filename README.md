# Student Performance Grade Prediction

Machine learning project for predicting students' final grade (`T3`) from academic, demographic, family and social variables. The project compares supervised regression models under two prediction settings: with and without prior term grades.

**Project date:** Academic year 2024/2025.

## Goal

The target is to predict `T3`, the final student grade, using the available student performance dataset.

Two modeling scenarios are evaluated:

- **Model 1: with prior grades.** Uses `T1`, `T2` and all other available predictors.
- **Model 2: without prior grades.** Excludes `T1` and `T2` to estimate performance when previous grades are not available.

## Repository Structure

```text
.
|-- imports/                              # Utility code from previous lab work
|-- src/
|   |-- additional_analysis.ipynb         # PCA, clustering and additional analysis
|   |-- baseline_models.py                # Custom baseline regressors
|   |-- exploratory_analysis.ipynb        # Initial exploratory data analysis
|   |-- generate_predictions.py           # Final prediction generation
|   |-- model_with_prior_grades.py        # Model 1 experiments
|   |-- model_without_prior_grades.py     # Model 2 experiments
|   `-- preprocessing.py                  # Data cleaning and feature engineering
|-- final_predictions.csv                 # Final predictions included with the project
|-- Informe_Proyecto_Machine_Learning.pdf # Original report in Spanish
`-- requirements.txt
```

## Models

### Model 1: With Prior Grades

- `T2Predictor`
- Linear Regression
- ElasticNet
- SVR
- Random Forest
- XGBoost

### Model 2: Without Prior Grades

- `MeanPredictor`
- KNN
- ElasticNet
- MLPRegressor
- Random Forest
- XGBoost

## Methodology

- Missing-value handling with median imputation.
- Categorical encoding with one-hot encoding.
- Feature engineering, including `T1_T2_interaction` and a combined alcohol-consumption feature.
- Model selection with 5-fold cross-validation.
- Evaluation using `R2`, MAE and MSE.
- Final prediction generation with the selected XGBoost configurations.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The scripts expect the original data files in a local `data/` directory:

```text
data/
|-- rendimiento_estudiantes_train.csv
`-- rendimiento_estudiantes_test_vacio.csv
```

After preprocessing, the following generated files are created:

```text
data/
|-- rendimiento_estudiantes_train_EDA.csv
`-- rendimiento_estudiantes_test_EDA.csv
```

The `data/` directory is intentionally ignored by Git to avoid publishing raw or generated datasets by accident.

## Usage

Run the project from the repository root:

```bash
python src/preprocessing.py
python src/model_with_prior_grades.py
python src/model_without_prior_grades.py
python src/generate_predictions.py
```

The final prediction file included in the repository is:

```text
final_predictions.csv
```

## Report

The original written report is kept in Spanish as `Informe_Proyecto_Machine_Learning.pdf`.
