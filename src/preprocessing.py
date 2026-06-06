import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import os

# Current file path
base_path = os.path.dirname(__file__)

# -------------------- Load and process training data --------------------
df = pd.read_csv(os.path.join(base_path, "../data/rendimiento_estudiantes_train.csv"))

# Define variables
categorical_vars = [
    'escuela', 'sexo', 'entorno', 'TamFam', 'EstPadres', 'razon', 'tutor',
    'apoyo', 'ApFam', 'academia', 'extras', 'enfermeria', 'EstSup',
    'internet', 'pareja', 'asignatura'
]
discrete_vars = [
    'edad', 'Medu', 'Pedu', 'TiempoViaje', 'TiempoEstudio', 'suspensos',
    'RelFam', 'TiempoLib', 'SalAm', 'AlcSem', 'AlcFin', 'salud'
]
continuous_vars = ['faltas', 'T1', 'T2', 'T3']

# Replacements and transformations
df['razon'] = df['razon'].replace('otras', 'otros')
df.loc[df['faltas'] > 100, 'faltas'] = np.nan

# Imputation
missing_cols = [col for col in df.columns if df[col].isnull().any()]
imputer = SimpleImputer(strategy='median')
df[missing_cols] = imputer.fit_transform(df[missing_cols])

# New features
df['T1_T2_interaction'] = df['T1'] * df['T2']
df['Alc_comb'] = (5/7) * df['AlcSem'] + (2/7) * df['AlcFin']

for ocup in ['docencia', 'otros', 'sanidad', 'servicios']:
    df[ocup] = ((df['Mtrab'] == ocup) | (df['Ptrab'] == ocup)).astype(int)

df.drop(columns=["AlcSem", "AlcFin", "Mtrab", "Ptrab"], inplace=True)
# Binary variables: transform yes/no values into 1/0
for col in categorical_vars[:]:
    if df[col].isin(['si', 'no']).all():
        df[col] = df[col].map({'si': 1, 'no': 0})
        categorical_vars.remove(col)

# One-hot encoding
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True).astype(int)

# Save processed training data
df.to_csv(os.path.join(base_path, "../data/rendimiento_estudiantes_train_EDA.csv"))

# -------------------- Process test data --------------------
df_test = pd.read_csv(os.path.join(base_path, "../data/rendimiento_estudiantes_test_vacio.csv"))

df_test['razon'] = df_test['razon'].replace('otras', 'otros')
df_test.loc[df_test['faltas'] > 100, 'faltas'] = np.nan


missing_cols = [col for col in df_test.columns if df_test[col].isnull().any()]
print("Columns with missing values:", missing_cols)
# No missing values

df_test['T1_T2_interaction'] = df_test['T1'] * df_test['T2']
df_test['Alc_comb'] = (5/7) * df_test['AlcSem'] + (2/7) * df_test['AlcFin']

for ocup in ['docencia', 'otros', 'sanidad', 'servicios']:
    df_test[ocup] = ((df_test['Mtrab'] == ocup) | (df_test['Ptrab'] == ocup)).astype(int)

df_test.drop(columns=["AlcSem", "AlcFin", "Mtrab", "Ptrab"], inplace=True)
# One-hot encoding with the same logic used for training data
for col in categorical_vars[:]:
    if df_test[col].isin(['si', 'no']).all():
        df_test[col] = df_test[col].map({'si': 1, 'no': 0})
        categorical_vars.remove(col)

df_test = pd.get_dummies(df_test, columns=categorical_vars, drop_first=True)

# Ensure test data has the same columns as training data
missing_cols = set(df.columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0

# Ensure columns are in the same order as training data
df_test = df[df_test.columns.intersection(df.columns)].reindex(columns=df.columns, fill_value=0)

df_test.to_csv(os.path.join(base_path, "../data/rendimiento_estudiantes_test_EDA.csv"))
