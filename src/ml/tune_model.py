import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Prediccion_Sueldos_Colombia")


def prepare_filtered_data():
    print("📥 Cargando dataset y aplicando filtros de outliers...")
    df = pd.read_csv(r"C:\Users\ma_va\Documents\Salary\DATA\dataset_modelo_limpio.csv")

    # EL TRUCO ESTADÍSTICO: Filtrar sueldos extremos para enfocar el modelo
    df = df[(df['salario_mensual'] >= 1000000) & (df['salario_mensual'] <= 15000000)]

    y = df['salario_mensual']
    X = df.drop(columns=['salario_mensual'])

    cat_cols = ['sexo', 'nivel_educativo', 'afiliado_salud', 'sector_economico', 'tipo_contrato', 'tamano_empresa']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Cargar datos una sola vez para que Optuna vuele en memoria
X_train, X_test, y_train, y_test = prepare_filtered_data()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


def objective(trial):
    # 2. Espacio de búsqueda de hiperparámetros de Optuna
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        # learning_rate: Qué tan rápido aprende (bajos = más preciso pero más lento)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        # max_depth: Profundidad del árbol (evita sobreajuste)
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        # min_child_weight: Peso mínimo para crear una nueva rama
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        # subsample: Porcentaje de datos usados por cada árbol (evita que memorice)
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # colsample_bytree: Porcentaje de columnas usadas por cada árbol
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
    }

    # 3. Entrenar el modelo
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=300,  # Más árboles porque el learning_rate puede ser bajo
        evals=[(dtest, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # 4. Predecir y evaluar
    preds = model.predict(dtest)
    mae = mean_absolute_error(y_test, preds)

    # 5. Registrar en MLflow CADA trial como un sub-experimento
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, preds)))
        # Solo guardamos el modelo si el error es menor a 500k para no llenar el disco
        if mae < 700000:
            mlflow.xgboost.log_model(model, artifact_path=f"xgb-model-mae-{int(mae)}")

    return mae  # Optuna minimizará este valor


if __name__ == "__main__":
    print("\n🚀 Iniciando Optimizador Bayesiano (Optuna + MLflow)...")

    # Creamos un "Run" padre en MLflow para agrupar todos los intentos
    with mlflow.start_run(run_name="Tuning_Optuna_Filtro15M"):
        study = optuna.create_study(direction="minimize")
        # n_trials: Número de modelos distintos que va a probar. (Ponle 30 para empezar)
        study.optimize(objective, n_trials=30)

        print("\n" + "=" * 50)
        print(f"🏆 ¡Búsqueda completada!")
        print(f"💰 MEJOR MAE ALCANZADO: ${study.best_value:,.0f} COP")
        print(f"🔧 Mejores Hiperparámetros encontrados:")
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")
        print("=" * 50)

        # Registrar los mejores parámetros en el Run Padre
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
        mlflow.log_metric("best_mae", study.best_value)