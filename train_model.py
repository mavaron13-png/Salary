import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Generar Datos Sintéticos (Reemplazar luego con tu dataset del DANE/Kaggle)
def load_data():
    np.random.seed(42)
    n_samples = 1000

    # Features: Años de experiencia, Nivel educativo (1-5), Sector (1-10)
    experiencia = np.random.randint(0, 20, n_samples)
    educacion = np.random.randint(1, 6, n_samples)
    sector = np.random.randint(1, 11, n_samples)

    # Target: Sueldo base + variables (con un poco de ruido)
    sueldo = 1500000 + (experiencia * 200000) + (educacion * 300000) + (sector * 50000) + np.random.normal(0, 200000,
                                                                                                           n_samples)

    X = pd.DataFrame({'experiencia': experiencia, 'educacion': educacion, 'sector': sector})
    y = sueldo
    return X, y


# 2. Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Asume que correrás el server localmente
mlflow.set_experiment("Prediccion_Sueldos_Colombia")


def train_and_tune():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir datos al formato nativo DMatrix de XGBoost (más rápido)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 3. Definir Hiperparámetros a probar (Tuning manual básico para el ejemplo)
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }

    n_estimators = 100

    # 4. Iniciar el run en MLflow
    with mlflow.start_run(run_name="XGBoost_Base_Model"):
        # Registrar hiperparámetros
        mlflow.log_params(params)
        mlflow.log_param("n_estimators", n_estimators)

        # Entrenar el modelo
        evals = [(dtrain, "train"), (dtest, "validation")]
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # 5. Predecir y calcular métricas
        preds = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        # Registrar métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # 6. Guardar el modelo en MLflow
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        print(f"Entrenamiento completado. RMSE: {rmse:,.2f} COP | MAE: {mae:,.2f} COP")


if __name__ == "__main__":
    train_and_tune()