# Proyecto Final - Aprendizaje Automatico

Proyecto final de la asignatura Aprendizaje Automatico 2024/2025. El objetivo es predecir la nota final de estudiantes (`T3`) a partir de variables academicas, demograficas y sociales mediante modelos de regresion supervisada.

## Objetivo

El proyecto compara dos escenarios de modelado:

- **Modelo 1:** predice `T3` usando `T1`, `T2` y el resto de variables disponibles.
- **Modelo 2:** predice `T3` sin usar `T1` ni `T2`, para evaluar el rendimiento cuando no se dispone de notas previas.

## Estructura

```text
.
|-- imports/                         # Funciones auxiliares usadas en practicas previas
|-- src/
|   |-- clases.py                    # Baselines personalizados
|   |-- exploracion.ipynb            # Analisis exploratorio inicial
|   |-- exploracion_adicional.ipynb  # PCA, clustering y analisis adicional
|   |-- modelo1.py                   # Modelos con T1 y T2
|   |-- modelo2.py                   # Modelos sin T1 ni T2
|   |-- prediccion.py                # Generacion de predicciones finales
|   `-- preprocesado.py              # Limpieza y transformacion de datos
|-- Informe_Proyecto_Machine_Learning.pdf
|-- predicciones_finales.csv
`-- requirements.txt
```

## Modelos

### Modelo 1: con notas previas

- `T2Predictor`
- Regresion lineal
- ElasticNet
- SVR
- Random Forest
- XGBoost

### Modelo 2: sin notas previas

- `MediaPredictor`
- KNN
- ElasticNet
- MLPRegressor
- Random Forest
- XGBoost

## Metodologia

- Limpieza e imputacion de valores ausentes.
- Transformacion de variables categoricas mediante one-hot encoding.
- Generacion de variables derivadas, como interaccion `T1_T2_interaction` y consumo combinado de alcohol.
- Comparacion de modelos mediante validacion cruzada de 5 folds.
- Evaluacion con `R2`, MAE y MSE.
- Entrenamiento final y generacion de predicciones.

## Instalacion

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Datos

Los scripts esperan los archivos originales en una carpeta `data/`:

```text
data/
|-- rendimiento_estudiantes_train.csv
`-- rendimiento_estudiantes_test_vacio.csv
```

Despues del preprocesado se generan:

```text
data/
|-- rendimiento_estudiantes_train_EDA.csv
`-- rendimiento_estudiantes_test_EDA.csv
```

La carpeta `data/` se mantiene fuera de Git para evitar publicar datos brutos o generados accidentalmente.

## Ejecucion

Desde la raiz del proyecto:

```bash
python src/preprocesado.py
python src/modelo1.py
python src/modelo2.py
python src/prediccion.py
```

El archivo final de predicciones incluido en este repositorio es:

```text
predicciones_finales.csv
```

## Informe

El documento `Informe_Proyecto_Machine_Learning.pdf` resume el planteamiento, el analisis exploratorio, los experimentos realizados y las conclusiones del proyecto.
