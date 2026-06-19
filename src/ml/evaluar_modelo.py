import os
import json
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error


def ejecutar_evaluacion():
    print("📦 [MLFLOW] Iniciando tracking de evaluación...")

    # 1. Configurar MLflow para usar tracking local en archivos (ideal para CI/CD)
    mlflow.set_tracking_uri("file:./mlflow_runs")
    mlflow.set_experiment("Evaluacion_Anual_Remuneracion")

    # Rutas relativas del proyecto
    ruta_modelo = "./DATA/xgboost_produccion.json"
    ruta_test_data = "./DATA/dataset_test.csv"  # Tu porción de datos de validación/test

    # Seguro por si no ha salido el dataset de test oficial (creamos un mock rápido para no romper el CI/CD)
    if not os.path.exists(ruta_test_data):
        print("⚠️ Dataset de prueba no encontrado. Generando una muestra sintética de evaluación...")
        os.makedirs("./DATA", exist_ok=True)
        # Generamos un DataFrame de prueba con la estructura exacta del modelo
        modelo_dummy = xgb.Booster()
        modelo_dummy.load_model(ruta_modelo)
        columnas = modelo_dummy.feature_names

        # Muestra ficticia de 100 registros para evaluar
        df_mock = pd.DataFrame(0, index=range(100), columns=columnas)
        df_mock["edad"] = 35
        df_mock["meses_experiencia"] = 48
        df_mock["horas_semanales"] = 40
        # Supongamos una columna objetivo 'salario' para medir error
        df_mock["salario"] = 4500000
        df_mock.to_csv(ruta_test_data, index=False)

    # 2. Cargar Datos y Modelo
    df_test = pd.read_csv(ruta_test_data)
    y_true = df_test["salario"]
    X_test = df_test.drop(columns=["salario"], errors="ignore")

    bst = xgb.Booster()
    bst.load_model(ruta_modelo)

    # Alinear las columnas por si el orden cambió
    X_test = X_test[bst.feature_names]
    dtest = xgb.DMatrix(X_test)

    # 3. Iniciar el Experimento en MLflow
    with mlflow.start_run(run_name="Evaluacion_Batch_Anual"):
        print("🚀 Ejecutando predicciones con XGBoost...")
        y_pred = bst.predict(dtest)

        # Calcular la métrica reina para regresión: RMSE
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"📊 RMSE Calculado: {rmse:,.2f}")

        # Registrar parámetros críticos del modelo en MLflow
        mlflow.log_param("algoritmo", "XGBoost")
        mlflow.log_param("num_features", len(bst.feature_names))

        # Registrar la métrica de performance
        mlflow.log_metric("rmse", rmse)

        # Guardar el modelo como un artefacto versionado dentro de la corrida
        mlflow.xgboost.log_model(bst, artifact_path="modelo_campeon")

        # Guardar un resumen rápido en JSON legible para el equipo
        resumen = {"rmse_produccion": float(rmse), "total_registros_evaluados": len(df_test)}
        with open("resumen_evaluacion.json", "w") as f:
            json.dump(resumen, f, indent=4)

        mlflow.log_artifact("resumen_evaluacion.json")

    print("✅ [MLFLOW] Evaluación registrada de manera exitosa en './mlflow_runs'")


if __name__ == "__main__":
    ejecutar_evaluacion()