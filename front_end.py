import gradio as gr
import PyPDF2
import json
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import HumanMessage

# Importamos a nuestro Agente (asegúrate de que exista en agente.py)
from orquestador import orquestador

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# El "Menú" estricto para el LLM y la interfaz
SECTORES_CIIU = {
    62: "Tecnología, Software y TI",
    64: "Finanzas y Seguros",
    69: "Legal y Contabilidad",
    71: "Ingeniería y Arquitectura",
    86: "Salud y Medicina",
    85: "Educación y Universidades",
    41: "Construcción e Inmobiliario",
    47: "Retail y Comercio al por menor",
    49: "Logística y Transporte",
    73: "Publicidad y Marketing",
    10: "Agroindustria y Alimentos",
    70: "Consultoría y Gestión Empresarial",
    84: "Administración Pública (Gobierno)",
    55: "Turismo y Hotelería",
    90: "Artes, Entretenimiento y Medios"
}


# --- FASE 1: EXTRACCIÓN CON LLM ---
# --- FASE 1: EXTRACCIÓN CON LLM (AHORA BLINDADA) ---
def extraer_datos_pdf(archivo_pdf):
    if archivo_pdf is None:
        return [gr.update()] * 6 + ["Sube un archivo primero."]

    try:
        print("📄 Leyendo PDF...")
        lector = PyPDF2.PdfReader(archivo_pdf.name)
        texto_perfil = ""
        for pagina in lector.pages:
            texto_perfil += pagina.extract_text()

        texto_perfil = texto_perfil[:4000]

        opciones_sectores = "\n".join([f"Código {k}: {v}" for k, v in SECTORES_CIIU.items()])

        # EL NUEVO PROMPT CON GUARDIA DE SEGURIDAD
        prompt_extraccion = f"""
        Eres un analizador estricto de perfiles de LinkedIn y currículums.

        REGLA DE ORO: Primero, determina si el texto proporcionado es realmente un perfil profesional, CV o resumen laboral.
        Si el texto es sobre recetas, manuales, cuentos, o cualquier cosa no relacionada con la trayectoria laboral de una persona, debes rechazarlo.

        Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura:

        Si NO es un perfil válido:
        {{
            "es_perfil_valido": false
        }}

        Si SÍ es un perfil válido:
        {{
            "es_perfil_valido": true,
            "edad": entero (asume 18 años al inicio de la universidad),
            "sexo": 1 para Hombre, 2 para Mujer,
            "nivel_educativo": 6 para pregrado, 7 para posgrado/maestría,
            "sector_economico": Elige el código numérico que mejor encaje de esta lista exacta:\n{opciones_sectores},
            "meses_experiencia": entero (suma de todo su tiempo laboral),
            "ultimo_cargo": El título exacto de su trabajo actual o más reciente
        }}

        Texto del perfil:
        {texto_perfil}
        """

        print("🧠 GPT-4o-mini evaluando y extrayendo datos...")
        respuesta = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt_extraccion}]
        )

        datos = json.loads(respuesta.choices[0].message.content)

        # 🛑 AQUÍ ESTÁ EL BLOQUEO: Si la IA dice que es falso, abortamos.
        if not datos.get("es_perfil_valido", True):
            print("🛑 [SEGURIDAD] PDF rechazado por no ser un perfil profesional.")
            return (
                gr.update(value=30), gr.update(value=1), gr.update(value=6),
                gr.update(value=""), gr.update(value=0), gr.update(value=""),
                "🛑 ERROR: El documento subido no parece ser un perfil de LinkedIn o CV válido. Por favor sube un documento laboral real."
            )

        # Si pasó el filtro, procesamos normalmente
        codigo_sector = datos.get("sector_economico", 62)
        nombre_sector = SECTORES_CIIU.get(codigo_sector, "Otro / No listado")
        sector_visual = f"{codigo_sector} - {nombre_sector}"

        print(f"✅ Datos extraídos: {datos}")

        return (
            datos.get("edad", 30),
            datos.get("sexo", 1),
            datos.get("nivel_educativo", 6),
            sector_visual,
            datos.get("meses_experiencia", 36),
            datos.get("ultimo_cargo", "Profesional"),
            "✅ Perfil extraído y validado con éxito. Revisa y completa los datos manuales."
        )
    except Exception as e:
        return [gr.update()] * 6 + [f"❌ Error al extraer: {str(e)}"]

# --- FASE 2: INFERENCIA DUAL ---
def calcular_salario_dual(edad, sexo, educacion, salud, sector_visual, contrato, exp, horas, tamano_empresa,
                          ultimo_cargo):
    # Extraemos el número del sector_visual (Ej. de "62 - Tecnología" sacamos el 62 entero)
    try:
        sector_num = int(str(sector_visual).split(" - ")[0])
    except:
        sector_num = 62  # Fallback por si acaso

    # 1. Petición a XGBoost
    url_fastapi = "http://127.0.0.1:8000/predict"
    payload = {
        "edad": edad, "sexo": sexo, "nivel_educativo": educacion,
        "afiliado_salud": salud, "sector_economico": sector_num,
        "tipo_contrato": contrato, "meses_experiencia": exp,
        "horas_semanales": horas, "tamano_empresa": tamano_empresa
    }

    try:
        print("🚀 Consultando XGBoost...")
        res = requests.post(url_fastapi, json=payload)
        if res.status_code == 200:
            data = res.json()
            resultado_ml = f"### 🤖 Modelo Matemático (XGBoost)\n- **Salario Estimado:** ${data['salario_estimado_cop']:,.2f} COP\n- **Rango Sugerido:** ${data['rango_sugerido_min_cop']:,.2f} - ${data['rango_sugerido_max_cop']:,.2f} COP"
        else:
            resultado_ml = f"Error en FastAPI: {res.text}"
    except Exception as e:
        resultado_ml = f"Error conectando al modelo: {e}"

    # 2. Petición al Agente RAG
    try:
        print("📚 Consultando RAG Documental...")
        mapa_tamano = {1: "empresa pequeña", 2: "empresa mediana", 3: "empresa grande"}
        str_tamano = mapa_tamano.get(tamano_empresa, "empresa")

        prompt_agente = f"Según los estudios de mercado, ¿cuánto gana un {ultimo_cargo} en una {str_tamano} en Colombia? Muestra los datos exactos que encuentres en los documentos."

        # Invocamos al Orquestador de LangGraph
        respuesta = orquestador.invoke({"messages": [HumanMessage(content=prompt_agente)]})
        # El Orquestador nos devuelve el estado final, tomamos el último mensaje
        resultado_rag = f"### 📚 Análisis del Agente\n{respuesta['messages'][-1].content}"
    except Exception as e:
        resultado_rag = f"Error en Agente RAG: {e}"

    return resultado_ml, resultado_rag


# --- INTERFAZ GRÁFICA CON GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Calculadora de Remuneración Profesional - IA Dual")
    gr.Markdown(
        "Sube tu perfil de LinkedIn en PDF. La IA extraerá tus datos y cruzará un modelo matemático predictivo con los estudios de mercado oficiales.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Sube tu Profile.pdf de LinkedIn", file_types=[".pdf"])
            btn_extraer = gr.Button("1️⃣ Autocompletar mi Perfil", variant="secondary")
            lbl_estado = gr.Textbox(label="Estado de Extracción", interactive=False)

            gr.Markdown("### 🤖 Datos Extraídos por la IA (Editables)")
            ultimo_cargo = gr.Textbox(label="Último Cargo (Se usa para buscar en Estudios de Mercado)")
            edad = gr.Slider(18, 70, value=30, step=1, label="Edad Inferida")
            sexo = gr.Radio(choices=[("Hombre", 1), ("Mujer", 2)], value=1, label="Sexo")
            educacion = gr.Dropdown(
                choices=[("Bachiller", 4), ("Técnico/Tecnológico", 5), ("Universitario (Pregrado)", 6),
                         ("Especialización/Maestría/Doctorado", 7)],
                value=6, label="Nivel Educativo"
            )

            # Ahora el sector es de texto y muestra la descripción
            sector_visual = gr.Textbox(label="Sector Económico Inferido (CIIU)", interactive=True)
            exp = gr.Number(value=36, label="Meses de Experiencia Total")

        with gr.Column():
            gr.Markdown("### ✍️ Datos Manuales (Por favor completa)")

            # NUEVAS ETIQUETAS SOLICITADAS
            horas = gr.Slider(10, 60, value=40, step=1, label="Horas semanales de trabajo usuales")
            salud = gr.Radio(choices=[("Sí", 1), ("No", 2)], value=1, label="¿Está afiliado a salud?")
            contrato = gr.Dropdown(
                choices=[("Término Indefinido", 1), ("Prestación de Servicios", 2), ("Obra/Labor", 3)], value=1,
                label="Último tipo de contrato")
            tamano_empresa = gr.Dropdown(
                choices=[("Pequeña (<50 empleados)", 1), ("Mediana (50-200)", 2), ("Grande (>200)", 3)], value=3,
                label="¿Cuál es el tamaño de la empresa?")

            btn_calcular = gr.Button("2️⃣ Calcular mi Salario Ideal", variant="primary")

    with gr.Row():
        out_ml = gr.Markdown()
        out_rag = gr.Markdown()

    # Eventos
    btn_extraer.click(
        fn=extraer_datos_pdf,
        inputs=[pdf_input],
        outputs=[edad, sexo, educacion, sector_visual, exp, ultimo_cargo, lbl_estado]
    )

    btn_calcular.click(
        fn=calcular_salario_dual,
        inputs=[edad, sexo, educacion, salud, sector_visual, contrato, exp, horas, tamano_empresa, ultimo_cargo],
        outputs=[out_ml, out_rag]
    )

if __name__ == "__main__":
    demo.launch(share=False)