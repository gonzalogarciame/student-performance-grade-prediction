import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor


from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports.lab_2_3_LinearRegression import LinearRegressor
from imports.Lab2_6_CV import cross_validation
from sklearn.model_selection import train_test_split

from baseline_models import T2Predictor

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/rendimiento_estudiantes_train_EDA.csv"), index_col=0)

    X1 = df.drop(columns=["T3"]).values
    y = df["T3"].values

    X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2,random_state=42)

    print("MODEL 1: WITH PRIOR GRADES")
    # -----------------------
    # T2Predictor
    # -----------------------
    T2_model = T2Predictor(df.drop(columns=["T3"]).columns.tolist())
    T2_model.fit(X1_train, y_train)

    mean_score, std_score = cross_validation(T2_model, X1_train, y_train, nFolds=5)
    print("\n=== T2Predictor Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_T2 = T2_model.predict(X1_test)

    test_mae_T2 = mean_absolute_error(y_test, y_test_pred_T2)
    test_mse_T2 = mean_squared_error(y_test, y_test_pred_T2)
    test_r2_T2 = r2_score(y_test, y_test_pred_T2)

    print("=== T2Predictor Test Evaluation ===")
    print(f"Test MAE: {test_mae_T2:.5f}")
    print(f"Test MSE: {test_mse_T2:.5f}")
    print(f"Test R²: {test_r2_T2:.5f}\n")


    # -----------------------
    # Linear Regression
    # -----------------------
    linear_model = LinearRegressor()
    linear_model.fit(X1_train, y_train)

    mean_score, std_score = cross_validation(linear_model, X1_train, y_train, nFolds=5)
    print("\n=== LinearRegression Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_lr = linear_model.predict(X1_test)

    test_mae_lr = mean_absolute_error(y_test, y_test_pred_lr)
    test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
    test_r2_lr = r2_score(y_test, y_test_pred_lr)

    print("=== LinearRegression Test Evaluation ===")
    print(f"Test MAE: {test_mae_lr:.5f}")
    print(f"Test MSE: {test_mse_lr:.5f}")
    print(f"Test R²: {test_r2_lr:.5f}\n")

    # -----------------------
    # ElasticNet
    # -----------------------
    param_grid_en = {
        'alpha': np.logspace(-3, 2, 10),
        'l1_ratio': np.arange(0.1, 1.1, 0.1)
    }

    grid_en = GridSearchCV(ElasticNet(random_state=42, max_iter=5000),
                        param_grid=param_grid_en,
                        scoring='r2',
                        cv=5,
                        n_jobs=-1)
    grid_en.fit(X1_train, y_train)
    best_en = grid_en.best_estimator_

    mean_score, std_score = cross_validation(best_en, X1_train, y_train, nFolds=5)
    print("\n=== ElasticNet Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_en = best_en.predict(X1_test)

    test_mae_en = mean_absolute_error(y_test, y_test_pred_en)
    test_mse_en = mean_squared_error(y_test, y_test_pred_en)
    test_r2_en = r2_score(y_test, y_test_pred_en)

    print("=== ElasticNet Test Evaluation ===")
    print(f"Test MAE: {test_mae_en:.5f}")
    print(f"Test MSE: {test_mse_en:.5f}")
    print(f"Test R²: {test_r2_en:.5f}\n")

    print(f"Best ElasticNet parameters: {grid_en.best_params_}")

    # -----------------------
    # SVR (con GridSearchCV)
    # -----------------------
    param_grid_svr = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_svr = GridSearchCV(SVR(), param_grid=param_grid_svr, scoring='r2', cv=5, n_jobs=-1)
    grid_svr.fit(X1_train, y_train)
    best_svr = grid_svr.best_estimator_

    mean_score, std_score = cross_validation(best_svr, X1_train, y_train, nFolds=5)
    print("\n=== SVR Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_svr = best_svr.predict(X1_test)

    test_mae_svr = mean_absolute_error(y_test, y_test_pred_svr)
    test_mse_svr = mean_squared_error(y_test, y_test_pred_svr)
    test_r2_svr = r2_score(y_test, y_test_pred_svr)

    print("=== SVR Test Evaluation ===")
    print(f"Test MAE: {test_mae_svr:.5f}")
    print(f"Test MSE: {test_mse_svr:.5f}")
    print(f"Test R²: {test_r2_svr:.5f}\n")



    # -----------------------
    # RandomForestRegressor
    # -----------------------
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                        param_grid=param_grid_rf,
                        scoring='r2',
                        cv=5,
                        n_jobs=-1)
    grid_rf.fit(X1_train, y_train)
    best_rf = grid_rf.best_estimator_

    mean_score, std_score = cross_validation(best_rf, X1_train, y_train, nFolds=5)
    print("\n=== RandomForest Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_rf = best_rf.predict(X1_test)

    test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
    test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)

    print("=== RandomForest Test Evaluation ===")
    print(f"Test MAE: {test_mae_rf:.5f}")
    print(f"Test MSE: {test_mse_rf:.5f}")
    print(f"Test R²: {test_r2_rf:.5f}\n")


    # -----------------------
    # XGBoost
    # -----------------------
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    }

    grid_xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                            param_grid=param_grid_xgb,
                            scoring='r2',
                            cv=5,
                            n_jobs=-1)
    grid_xgb.fit(X1_train, y_train)
    best_xgb1 = grid_xgb.best_estimator_

    mean_score, std_score = cross_validation(best_xgb1, X1_train, y_train, nFolds=5)
    print("\n=== XGBoost Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_xgb = best_xgb1.predict(X1_test)

    test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
    test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)
    test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

    print("=== XGBoost Test Evaluation ===")
    print(f"Test MAE: {test_mae_xgb:.5f}")
    print(f"Test MSE: {test_mse_xgb:.5f}")
    print(f"Test R²: {test_r2_xgb:.5f}\n")
