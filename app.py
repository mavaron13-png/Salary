from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import os
import logging
import traceback

# Configurar el logger
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Remuneración Profesional - Colombia",
    description="Backend en FastAPI para predecir salarios.",
    version="1.0.0"
)

# 1. Cargar el Modelo y extraer su molde de memoria
DATA_DIR = r"C:\Users\ma_va\Documents\Salary\DATA"
MODEL_PATH = os.path.join(DATA_DIR, "xgboost_produccion.json")

print("📥 Cargando modelo de producción desde disco local...")
try:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    # 🚨 LA MAGIA: Le pedimos a XGBoost su propia lista de columnas
    COLUMNAS_MODELO = model.feature_names
    print(f"✅ Modelo XGBoost cargado. Espera exactamente {len(COLUMNAS_MODELO)} características.")
except Exception as e:
    print(f"❌ Error al cargar el modelo local: {e}")
    model = None
    COLUMNAS_MODELO = []


# 2. Esquema de datos
class PerfilProfesional(BaseModel):
    edad: int
    sexo: int
    nivel_educativo: int
    afiliado_salud: int
    sector_economico: int
    tipo_contrato: int
    meses_experiencia: float
    horas_semanales: float
    tamano_empresa: int


@app.get("/")
def home():
    return {"status": "ok", "message": "API de Predicción de Salarios activa y escuchando."}


# 3. Endpoint de Inferencia
@app.post("/predict")
def predict_salary(perfil: PerfilProfesional):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        # 1. Parsear datos
        input_data = perfil.dict()
        df_input = pd.DataFrame([input_data])

        # 2. Dummies
        cat_cols = ['sexo', 'nivel_educativo', 'afiliado_salud', 'sector_economico', 'tipo_contrato', 'tamano_empresa']
        df_input_dummies = pd.get_dummies(df_input, columns=cat_cols)

        # 3. Alinear con el molde perfecto de XGBoost y forzar tipos
        df_aligned = df_input_dummies.reindex(columns=COLUMNAS_MODELO, fill_value=0)
        df_aligned = df_aligned.astype(float)

        # 4. Inferencia XGBoost
        dmatrix_input = xgb.DMatrix(df_aligned)
        prediction = model.predict(dmatrix_input)
        salario_predicho = float(prediction[0])

        return {
            "perfil_procesado": input_data,
            "salario_estimado_cop": round(salario_predicho, 2),
            "rango_sugerido_min_cop": round(max(0, salario_predicho - 652000), 2),
            "rango_sugerido_max_cop": round(salario_predicho + 652000, 2)
        }

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"\n{'=' * 50}\n❌ EXPLOSIÓN EN INFERENCIA:\n{error_traceback}\n{'=' * 50}")
        raise HTTPException(status_code=400, detail=f"Error interno: {str(e)}")