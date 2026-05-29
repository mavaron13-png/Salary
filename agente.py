import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 🔒 Cargar las variables ocultas del archivo .env
load_dotenv()


# 1. Definir el esquema estricto con Pydantic para el LLM
class PerfilSalarialInput(BaseModel):
    edad: int = Field(description="Edad en años del usuario.")
    sexo: int = Field(description="1 para Hombre, 2 para Mujer.")
    nivel_educativo: int = Field(description="Código DANE numérico (ej: 6 para Universitaria, 7 para Especialización).")
    afiliado_salud: int = Field(description="1 para Sí, 2 para No.")
    sector_economico: int = Field(description="Código numérico de la industria (ej: 62 para TI).")
    tipo_contrato: int = Field(description="Código numérico del contrato (ej: 1 para Indefinido).")
    meses_experiencia: float = Field(description="Meses de antigüedad o experiencia.")
    horas_semanales: float = Field(description="Horas de trabajo a la semana.")
    tamano_empresa: int = Field(description="Código del tamaño de empresa (ej: 3 para más de 50 empleados).")


# 2. Inyectar el esquema a la herramienta usando args_schema
@tool(args_schema=PerfilSalarialInput)
def consultar_modelo_salarial(edad: int, sexo: int, nivel_educativo: int, afiliado_salud: int,
                              sector_economico: int, tipo_contrato: int, meses_experiencia: float,
                              horas_semanales: float, tamano_empresa: int) -> str:
    """
    Útil para consultar la API de Machine Learning y predecir el salario de un profesional en Colombia.
    """
    url = "http://127.0.0.1:8000/predict"

    payload = {
        "edad": edad,
        "sexo": sexo,
        "nivel_educativo": nivel_educativo,
        "afiliado_salud": afiliado_salud,
        "sector_economico": sector_economico,
        "tipo_contrato": tipo_contrato,
        "meses_experiencia": meses_experiencia,
        "horas_semanales": horas_semanales,
        "tamano_empresa": tamano_empresa
    }

    try:
        response = requests.post(url, json=payload)
        # Si el servidor responde con error, lanzamos la excepción y capturamos el texto exacto
        response.raise_for_status()
        data = response.json()

        salario = data["salario_estimado_cop"]
        rango_min = data["rango_sugerido_min_cop"]
        rango_max = data["rango_sugerido_max_cop"]

        return f"El modelo predice un salario de ${salario:,.2f} COP. Rango sugerido: ${rango_min:,.2f} - ${rango_max:,.2f} COP."

    except requests.exceptions.RequestException as e:
        # Esto le devolverá el error detallado de FastAPI al LLM para que entienda por qué falló
        detalles = response.text if response is not None else str(e)
        return f"Error en la API: {detalles}"


tools = [consultar_modelo_salarial]

# 3. Configurar el LLM (OpenAI - GPT-4o-mini)
print("🧠 Inicializando el cerebro del Agente (GPT-4o-mini de OpenAI)...")
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("¡Peligro! No se encontró la OPENAI_API_KEY en el archivo .env")

    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.1
    )

    agente_laboral = create_react_agent(llm, tools)
    print("✅ Agente ReAct construido y listo para pensar.")

except Exception as e:
    print(f"❌ Error al inicializar LangGraph o OpenAI: {e}")

if __name__ == "__main__":
    system_prompt = """Eres un asistente experto en el mercado laboral colombiano. 
    Tu objetivo es ayudar a los usuarios a conocer su remuneración justa. 
    SIEMPRE usa la herramienta 'consultar_modelo_salarial' cuando el usuario te dé su perfil profesional. 
    Si te hacen preguntas fuera del ámbito laboral, declina amablemente la respuesta."""

    mensaje_usuario = """
    Hola, soy un hombre de 28 años. Tengo educación superior (código 6). 
    Trabajo en el sector de TI (código 62) con un contrato a término indefinido (código 1). 
    Llevo 36 meses en mi empresa actual, trabajo 40 horas a la semana y mi empresa tiene más de 50 empleados (código 3). 
    Sí estoy afiliado a salud (código 1). ¿Cuánto debería estar ganando según el mercado?
    """

    print("\n👤 Usuario:", mensaje_usuario)
    print("🤖 Agente pensando (esto puede tardar unos segundos)...\n")

    try:
        resultado = agente_laboral.invoke({
            "messages": [
                ("system", system_prompt),
                ("user", mensaje_usuario)
            ]
        })
        print(f"✅ Respuesta final del Agente:\n{resultado['messages'][-1].content}")
    except Exception as e:
        print(f"❌ El agente falló al intentar pensar: {e}")