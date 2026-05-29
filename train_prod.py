import pandas as pd
import xgboost as xgb
import os

DATA_DIR = r"C:\Users\ma_va\Documents\Salary\DATA"

print("📥 Cargando dataset para el entrenamiento final en Producción...")
df = pd.read_csv(os.path.join(DATA_DIR, "dataset_modelo_limpio.csv"))

# Filtro de outliers
df = df[(df['salario_mensual'] >= 1000000) & (df['salario_mensual'] <= 15000000)]

y = df['salario_mensual']
X = df.drop(columns=['salario_mensual'])

cat_cols = ['sexo', 'nivel_educativo', 'afiliado_salud', 'sector_economico', 'tipo_contrato', 'tamano_empresa']
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 1. 🚨 PON AQUÍ LA RECETA SECRETA QUE TE DIO OPTUNA 🚨
best_params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "learning_rate": 0.10558550006095731,       # Cambia por tu valor real
    "max_depth": 7,             # Cambia por tu valor real
    "min_child_weight": 7,      # Cambia por tu valor real
    "subsample": 0.9111834404797459,          # Cambia por tu valor real
    "colsample_bytree": 0.6127502788168343     # Cambia por tu valor real
}

print("🌳 Entrenando el modelo definitivo con todos los datos...")
# Usamos TODOS los datos (sin split de test) porque ya validamos que la receta funciona
dtrain = xgb.DMatrix(X, label=y)

model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=300 # Cantidad de árboles
)

# 2. GUARDAR DIRECTO EN DISCO (Sin depender de MLflow)
model_path = os.path.join(DATA_DIR, "xgboost_produccion.json")
model.save_model(model_path)

print(f"✅ ¡Modelo de producción guardado exitosamente en:\n{model_path}")