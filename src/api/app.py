from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import PyPDF2
import json
import os
import io
import xgboost as xgb
from openai import OpenAI
import pandas as pd

# Importamos nuestro orquestador
from src.agent.orquestador import orquestador
from langchain_core.messages import HumanMessage

app = FastAPI(title="API Calculadora de Remuneración IA")

# Permitir conexiones CORS (por si abres el HTML directo en el navegador)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el Modelo y Clientes
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
modelo_xgb = xgb.Booster()
try:
    modelo_xgb.load_model("data/models/xgboost_produccion.json")
    # Extraemos el nombre exacto de las columnas que espera el modelo
    columnas_modelo = modelo_xgb.feature_names
except Exception as e:
    print(f"⚠️ Error cargando XGBoost: {e}")
    columnas_modelo = []

SECTORES_CIIU = {
    62: "Tecnología, Software y TI", 64: "Finanzas y Seguros", 69: "Legal y Contabilidad",
    71: "Ingeniería y Arquitectura", 86: "Salud y Medicina", 85: "Educación y Universidades",
    41: "Construcción e Inmobiliario", 47: "Retail y Comercio al por menor", 49: "Logística y Transporte",
    73: "Publicidad y Marketing", 10: "Agroindustria y Alimentos", 70: "Consultoría y Gestión Empresarial",
    84: "Administración Pública (Gobierno)", 55: "Turismo y Hotelería", 90: "Artes, Entretenimiento y Medios"
}


# Modelos Pydantic para los datos de entrada
class DatosInferencia(BaseModel):
    edad: int
    sexo: int
    nivel_educativo: int
    afiliado_salud: int
    sector_economico: int
    tipo_contrato: int
    meses_experiencia: int
    horas_semanales: int
    tamano_empresa: int
    ultimo_cargo: str


# --- ENDPOINTS DE LA API ---

@app.post("/api/extraer-pdf")
async def extraer_pdf(file: UploadFile = File(...)):
    """Recibe un PDF de LinkedIn, extrae el texto y usa LLM para inferir variables."""
    try:
        contenido = await file.read()
        lector = PyPDF2.PdfReader(io.BytesIO(contenido))
        texto_perfil = "".join([pagina.extract_text() for pagina in lector.pages])[:4000]

        opciones_sectores = "\n".join([f"Código {k}: {v}" for k, v in SECTORES_CIIU.items()])

        prompt = f"""
        Eres un analizador estricto de perfiles de LinkedIn. Determina si el texto es un perfil profesional válido.
        Devuelve ÚNICAMENTE JSON.
        Si NO es válido: {{"es_perfil_valido": false}}
        Si SÍ es válido: 
        {{
            "es_perfil_valido": true, "edad": entero, "sexo": (1 Hombre, 2 Mujer),
            "nivel_educativo": (4 Bachiller, 5 Tecnico, 6 Pregrado, 7 Posgrado),
            "sector_economico": (elige el código de la lista: {opciones_sectores}),
            "meses_experiencia": entero, "ultimo_cargo": string
        }}
        Texto: {texto_perfil}
        """

        respuesta = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )

        datos = json.loads(respuesta.choices[0].message.content)
        if not datos.get("es_perfil_valido", True):
            raise HTTPException(status_code=400, detail="El PDF no es un CV válido.")

        datos["nombre_sector"] = SECTORES_CIIU.get(datos.get("sector_economico"), "Otro")
        return datos

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calcular-dual")
async def calcular_dual(datos: DatosInferencia):
    """Cruza XGBoost y RAG Agentico en una sola llamada."""
    resultado = {"xgboost": {}, "rag": {}}

    # 1. Inferencia del Modelo Matemático (XGBoost)
    try:
        df_completo = pd.DataFrame(0, index=[0], columns=columnas_modelo)
        df_input = pd.DataFrame([datos.dict(exclude={"ultimo_cargo"})])
        for col in df_input.columns:
            if col in df_completo.columns:
                df_completo.at[0, col] = df_input.at[0, col]

        col_sector = f"sector_economico_{datos.sector_economico}"
        if col_sector in df_completo.columns:
            df_completo.at[0, col_sector] = 1

        dmatrix = xgb.DMatrix(df_completo)
        salario_pred = float(modelo_xgb.predict(dmatrix)[0])

        resultado["xgboost"] = {
            "salario_estimado": salario_pred,
            "rango_min": salario_pred * 0.85,
            "rango_max": salario_pred * 1.15
        }
    except Exception as e:
        resultado["xgboost"]["error"] = str(e)
        salario_pred = 0

        # 2. Inferencia y Recuperación del RAG Agentico con Mapeo Estructurado
    try:
        mapa_tamano = {1: "empresa pequeña", 2: "empresa mediana", 3: "empresa grande"}
        str_tamano = mapa_tamano.get(datos.tamano_empresa, "empresa")
        nombre_sector = SECTORES_CIIU.get(datos.sector_economico, "Tecnología")

        instruccion_sector = f"en el sector '{nombre_sector}'"
        if datos.sector_economico == 62:
            instruccion_sector += " (INSTRUCCIÓN: Prioriza y mapea únicamente roles de ingeniería de software/desarrollo)."

        cargo_usuario = datos.ultimo_cargo  # Ejemplo: "Senior Software Developer"

        prompt_agente = f"""
        Actúa como un experto analista de compensación salarial en Colombia. 
        El usuario tiene el cargo de: '{cargo_usuario}'.
        La empresa es una {str_tamano} {instruccion_sector}.

        INSTRUCCIONES DE ALINEACIÓN SEMÁNTICA:
        1. Encuentra en los documentos el cargo semánticamente más idéntico al del usuario ('{cargo_usuario}') para rellenar el campo "cargo_identico_top1". EXCLUYE cargos directivos como CEO/Gerente General si el usuario es Developer.
        2. Genera una lista de 3 o 4 posiciones/retrievals relacionados del mercado laboral colombiano que sirvan como benchmark.
        3. Para cada elemento de la lista debes extraer: Cargo, Tag/Especialidad, Salario Mínimo Mensual (en número), Salario Máximo Mensual (en número) y Fuente (LHH o MyDNA).

        DEVES DEVOLVER TU RESPUESTA ESTRICTAMENTE EN ESTE FORMATO JSON (Sin texto afuera):
        {{
            "cargo_identico_top1": "Ej: Senior Software Developer / Ingeniero de Software Senior",
            "retrievals_mercado": [
                {{
                    "role": "Consultor Senior / Especialista",
                    "tag": "Tecnología & Digital · Ingeniería",
                    "min": 11500000,
                    "max": 16000000,
                    "source": "LHH"
                }},
                {{
                    "role": "Desarrollador Backend .NET Senior",
                    "tag": "Tecnología & Digital · Desarrollo",
                    "min": 12000000,
                    "max": 17500000,
                    "source": "MyDNA"
                }}
            ],
            "salario_minimo_absoluto": un_numero_entero_con_el_valor_mas_bajo_encontrado,
            "nota_reconciliacion": "💡 **Nota de Reconciliación:** Si aplica..."
        }}
        """

        respuesta_rag = orquestador.invoke({"messages": [HumanMessage(content=prompt_agente)]})
        texto_crudo = respuesta_rag['messages'][-1].content

        if "```json" in texto_crudo:
            texto_crudo = texto_crudo.split("```json")[1].split("```")[0].strip()
        elif "```" in texto_crudo:
            texto_crudo = texto_crudo.split("```")[1].split("```")[0].strip()

        data_json = json.loads(texto_crudo)

        resultado["rag"] = {
            "cargo_top1": data_json.get("cargo_identico_top1", cargo_usuario),
            "retrievals": data_json.get("retrievals_mercado", []),
            "valor_conservador": int(data_json.get("salario_minimo_absoluto", 11500000)),
            "nota": data_json.get("nota_reconciliacion", "")
        }
    except Exception as e:
        print(f"⚠️ Error en RAG estructurado: {e}")
        resultado["rag"] = {
            "cargo_top1": cargo_usuario,
            "retrievals": [
                {"role": "Senior Software Developer", "tag": "Tecnología · Desarrollo", "min": 11500000,
                 "max": 16000000, "source": "LHH"}
            ],
            "valor_conservador": 11500000,
            "nota": ""
        }

    return resultado



# --- SERVIR EL FRONTEND ESTÁTICO ---
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")


@app.get("/")
async def serve_index():
    return FileResponse("src/frontend/static/index.html")