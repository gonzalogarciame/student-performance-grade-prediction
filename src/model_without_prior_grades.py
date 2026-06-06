import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet

from xgboost import XGBRegressor

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports.Lab2_6_CV import cross_validation

from baseline_models import MeanPredictor


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/rendimiento_estudiantes_train_EDA.csv"), index_col=0)

    X2 = df.drop(columns=["T1","T2","T3", "T1_T2_interaction"]).values
    y = df["T3"].values

    X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2 ,random_state=42)




    print("MODEL 2: WITHOUT PRIOR GRADES") 

    #--------------
    # First model: MeanPredictor
    #--------------
    media_model = MeanPredictor()
    media_model.fit(X2_train, y_train)

    mean_score, std_score = cross_validation(media_model, X2_train, y_train, nFolds=5)
    print("\n=== MeanPredictor Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_media = media_model.predict(X2_test)

    test_mae_media = mean_absolute_error(y_test, y_test_pred_media)
    test_mse_media = mean_squared_error(y_test, y_test_pred_media)
    test_r2_media = r2_score(y_test, y_test_pred_media)

    print("=== MeanPredictor Test Evaluation ===")
    print(f"Test MAE: {test_mae_media:.5f}")
    print(f"Test MSE: {test_mse_media:.5f}")
    print(f"Test R²: {test_r2_media:.5f}\n")


    #--------------
    # Second model: KNN
    #--------------
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    knn_model = KNeighborsRegressor()
    grid_knn = GridSearchCV(
        estimator=knn_model,
        param_grid=param_grid_knn,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    grid_knn.fit(X2_train, y_train)
    best_knn = grid_knn.best_estimator_

    mean_score, std_score = cross_validation(best_knn, X2_train, y_train, nFolds=5)
    print("\n=== KNN Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_knn = best_knn.predict(X2_test)

    test_mae_knn = mean_absolute_error(y_test, y_test_pred_knn)
    test_mse_knn = mean_squared_error(y_test, y_test_pred_knn)
    test_r2_knn = r2_score(y_test, y_test_pred_knn)

    print("=== KNN Test Evaluation ===")
    print(f"Test MAE: {test_mae_knn:.5f}")
    print(f"Test MSE: {test_mse_knn:.5f}")
    print(f"Test R²: {test_r2_knn:.5f}\n")

    #--------------
    # Third model: ElasticNet
    #--------------
    param_grid_en = {
        'alpha': np.logspace(-3, 2, 10),
        'l1_ratio': np.arange(0.1, 1.1, 0.1)
    }

    en_model = ElasticNet(random_state=42, max_iter=5000)

    grid_en = GridSearchCV(
        estimator=en_model,
        param_grid=param_grid_en,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    grid_en.fit(X2_train, y_train)
    best_en = grid_en.best_estimator_

    mean_score, std_score = cross_validation(best_en, X2_train, y_train, nFolds=5)
    print("\n=== ElasticNet Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_en = best_en.predict(X2_test)

    test_mae_en = mean_absolute_error(y_test, y_test_pred_en)
    test_mse_en = mean_squared_error(y_test, y_test_pred_en)
    test_r2_en = r2_score(y_test, y_test_pred_en)

    print("=== ElasticNet Test Evaluation ===")
    print(f"Test MAE: {test_mae_en:.5f}")
    print(f"Test MSE: {test_mse_en:.5f}")
    print(f"Test R²: {test_r2_en:.5f}\n")

    print(f"Best ElasticNet parameters: {grid_en.best_params_}")

    #--------------
    # Fourth model: MLP
    #--------------
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'activation': ['relu', 'tanh'],
        'solver': ['adam']
    }

    mlp_model = MLPRegressor(random_state=42, max_iter=5000)
    grid_mlp = GridSearchCV(
        estimator=mlp_model,
        param_grid=param_grid_mlp,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    grid_mlp.fit(X2_train, y_train)
    best_mlp = grid_mlp.best_estimator_

    mean_score, std_score = cross_validation(best_mlp, X2_train, y_train, nFolds=5)
    print("\n=== MLP Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_mlp = best_mlp.predict(X2_test)

    test_mae_mlp = mean_absolute_error(y_test, y_test_pred_mlp)
    test_mse_mlp = mean_squared_error(y_test, y_test_pred_mlp)
    test_r2_mlp = r2_score(y_test, y_test_pred_mlp)

    print("=== MLP Test Evaluation ===")
    print(f"Test MAE: {test_mae_mlp:.5f}")
    print(f"Test MSE: {test_mse_mlp:.5f}")
    print(f"Test R²: {test_r2_mlp:.5f}\n")



    #--------------
    # Fifth model: RandomForest
    #--------------
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_model = RandomForestRegressor(random_state=42)

    grid_rf = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid_rf,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    grid_rf.fit(X2_train, y_train)
    best_rf = grid_rf.best_estimator_

    mean_score, std_score = cross_validation(best_rf, X2_train, y_train, nFolds=5)
    print("\n=== RandomForest Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_rf = best_rf.predict(X2_test)

    test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
    test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)

    print("=== RandomForest Test Evaluation ===")
    print(f"Test MAE: {test_mae_rf:.5f}")
    print(f"Test MSE: {test_mse_rf:.5f}")
    print(f"Test R²: {test_r2_rf:.5f}\n")


    #--------------
    # Sixth model: XGBoost
    #--------------
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    xgb_model = XGBRegressor(random_state=42)

    grid_xgb = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid_xgb,
        scoring='r2',
        cv=5,
        n_jobs=-1
    )

    grid_xgb.fit(X2_train, y_train)
    best_xgb2 = grid_xgb.best_estimator_

    mean_score, std_score = cross_validation(best_xgb2, X2_train, y_train, nFolds=5)
    print("\n=== XGBoost Cross Validation ===")
    print(f"R²: {mean_score:.5f} +- {std_score:.5f}")

    y_test_pred_xgb = best_xgb2.predict(X2_test)

    test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
    test_mse_xgb = mean_squared_error(y_test, y_test_pred_xgb)
    test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

    print("=== XGBoost Test Evaluation ===")
    print(f"Test MAE: {test_mae_xgb:.5f}")
    print(f"Test MSE: {test_mse_xgb:.5f}")
    print(f"Test R²: {test_r2_xgb:.5f}\n")
