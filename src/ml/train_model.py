import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Asegúrate de correr 'mlflow ui' en la terminal
mlflow.set_experiment("Prediccion_Sueldos_Colombia")


def prepare_real_data():
    print("📥 Cargando dataset limpio...")
    df = pd.read_csv(r"C:\Users\ma_va\Documents\Salary\DATA\dataset_modelo_limpio.csv")

    # 2. Separar el Target (Y) de las Features (X)
    y = df['salario_mensual']
    X = df.drop(columns=['salario_mensual'])

    # 3. One-Hot Encoding para las variables categóricas nominales
    # Le decimos explícitamente a Pandas qué columnas son categorías, aunque tengan números
    cat_cols = ['sexo', 'nivel_educativo', 'afiliado_salud', 'sector_economico', 'tipo_contrato', 'tamano_empresa']

    print("🔠 Aplicando One-Hot Encoding a variables categóricas...")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def train_and_tune():
    X, y = prepare_real_data()

    print("✂️ Separando en Train y Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir datos al formato nativo DMatrix de XGBoost (optimizado para velocidad y memoria)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 4. Hiperparámetros base (Aquí es donde luego jugarás con MLflow para mejorar el modelo)
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }
    n_estimators = 200

    print("🚀 Iniciando entrenamiento con MLflow...")
    with mlflow.start_run(run_name="XGBoost_Real_Data_V1"):
        mlflow.log_params(params)
        mlflow.log_param("n_estimators", n_estimators)

        evals = [(dtrain, "train"), (dtest, "validation")]

        # Entrenar el modelo
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=15,
            verbose_eval=10  # Imprimirá el progreso cada 10 árboles
        )

        # 5. Predecir y calcular métricas
        print("📊 Calculando métricas en el set de Test...")
        preds = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Guardar el modelo entrenado
        mlflow.xgboost.log_model(model, artifact_path="xgboost-colombia-model")

        print("\n" + "=" * 50)
        print(f"✅ Entrenamiento completado!")
        print(f"💰 Error Medio Absoluto (MAE): ${mae:,.0f} COP")
        print(f"📈 Raíz del Error Cuadrático (RMSE): ${rmse:,.0f} COP")
        print("=" * 50)


if __name__ == "__main__":
    train_and_tune()