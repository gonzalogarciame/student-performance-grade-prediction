import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# === Rutas ===
BASE_DIR = os.path.dirname(__file__)
train_path = os.path.join(BASE_DIR, "../data/rendimiento_estudiantes_train_EDA.csv")
test_path = os.path.join(BASE_DIR, "../data/rendimiento_estudiantes_test_EDA.csv")
output_path = os.path.join(BASE_DIR, "../data/predicciones_finales.csv")

# === Cargar datos ===
df = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

# === Preparar variables ===
X1 = df.drop(columns=["T3"]).values
X2 = df.drop(columns=["T1", "T2", "T3", "T1_T2_interaction"]).values
y = df["T3"].values

X1_train, _, y_train, _ = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, _, _, _ = train_test_split(X2, y, test_size=0.2, random_state=42)

X1_test = df_test.drop(columns=["T3"]).values
X2_test = df_test.drop(columns=["T1", "T2", "T1_T2_interaction", "T3"], errors="ignore").values

# === GridSearch para Modelo 1 ===
param_grid_xgb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

grid_xgb1 = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                         param_grid=param_grid_xgb,
                         scoring='r2',
                         cv=5,
                         n_jobs=-1)
grid_xgb1.fit(X1_train, y_train)
best_xgb1 = grid_xgb1.best_estimator_

# === GridSearch para Modelo 2 ===
grid_xgb2 = GridSearchCV(XGBRegressor(random_state=42, verbosity=0),
                         param_grid=param_grid_xgb,
                         scoring='r2',
                         cv=5,
                         n_jobs=-1)
grid_xgb2.fit(X2_train, y_train)
best_xgb2 = grid_xgb2.best_estimator_

# === Predicciones ===
pred1 = best_xgb1.predict(X1_test)
pred2 = best_xgb2.predict(X2_test)

# === Guardar predicciones ===
df_pred = pd.DataFrame({
    "Id": np.arange(1, len(df_test) + 1),
    "Modelo_1": pred1,
    "Modelo_2": pred2
})

df_pred.to_csv(output_path, index=False)
print("Archivo 'predicciones_finales.csv' generado correctamente.")