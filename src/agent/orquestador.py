import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import operator

# Importamos tu Agente Experto (el que ya armamos con XGBoost y RAG)
from src.agent.agente import agente_laboral

load_dotenv()
llm_rapido = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 1. Definimos la Memoria/Estado del Grafo
class EstadoGrafo(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tema_valido: bool


# 2. La "Camisa de Fuerza" del Guardia de Seguridad
class ClasificadorIntentos(BaseModel):
    es_laboral: bool = Field(
        description="True si el último mensaje del usuario es sobre salarios, perfiles, mercado laboral, empleos o empresas. False si es sobre recetas, política, clima o temas irrelevantes.")



# 3. NODO 1: El Guardia de Seguridad (Enrutador)
def nodo_guardia(estado: EstadoGrafo):
    print("🛡️ [LANGGRAPH] Evaluando intención del usuario...")
    mensajes = estado["messages"]
    ultimo_mensaje = mensajes[-1]

    # Robustez de Casteo: Verificamos si viene como tupla o como objeto
    if isinstance(ultimo_mensaje, tuple):
        texto_mensaje = ultimo_mensaje[1]  # Extrae el texto de ("user", "texto")
    else:
        texto_mensaje = ultimo_mensaje.content  # Extrae de un objeto HumanMessage

    # Usamos structured_output para garantizar un JSON perfecto
    llm_clasificador = llm_rapido.with_structured_output(ClasificadorIntentos)

    prompt = f"Evalúa si este mensaje requiere asistencia sobre el mercado laboral, sueldos o perfiles profesionales: '{texto_mensaje}'"
    resultado = llm_clasificador.invoke(prompt)

    print(f"🛡️ [LANGGRAPH] Decisión del Guardia -> Relevante: {resultado.es_laboral}")
    return {"tema_valido": resultado.es_laboral}


# 4. NODO 2: El Rechazo Rápido
def nodo_rechazo(estado: EstadoGrafo):
    print("🛑 [LANGGRAPH] Tema irrelevante detectado. Bloqueando acceso a herramientas.")
    respuesta = AIMessage(
        content="Lo siento, soy un asistente especializado **únicamente en el mercado laboral colombiano y remuneración profesional**. No puedo ayudarte con temas fuera de este ámbito. ¿Tienes alguna duda sobre salarios o perfiles?")
    return {"messages": [respuesta]}


# 5. NODO 3: Tu Agente ReAct Original
def nodo_agente_experto(estado: EstadoGrafo):
    print("✅ [LANGGRAPH] Tema aprobado. Despertando al Agente Experto...")
    # Le pasamos los mensajes al agente original que ya tiene las tools (RAG, FastAPI)
    respuesta = agente_laboral.invoke({"messages": estado["messages"]})

    # El Agente devuelve un dict con los nuevos mensajes, tomamos el último
    ultimo_mensaje_agente = respuesta["messages"][-1]
    return {"messages": [ultimo_mensaje_agente]}


# 6. Lógica Condicional (Los caminos del Grafo)
def ruta_condicional(estado: EstadoGrafo):
    if estado["tema_valido"]:
        return "ir_al_experto"
    else:
        return "ir_al_rechazo"


# --- CONSTRUCCIÓN DEL GRAFO ---
grafo = StateGraph(EstadoGrafo)

# Agregamos los nodos
grafo.add_node("Guardia", nodo_guardia)
grafo.add_node("Rechazo", nodo_rechazo)
grafo.add_node("Agente_Experto", nodo_agente_experto)

# Definimos el flujo
grafo.set_entry_point("Guardia")

# Agregamos la bifurcación
grafo.add_conditional_edges(
    "Guardia",
    ruta_condicional,
    {
        "ir_al_experto": "Agente_Experto",
        "ir_al_rechazo": "Rechazo"
    }
)

# Tanto el rechazo como el agente experto llevan al final de la interacción
grafo.add_edge("Rechazo", END)
grafo.add_edge("Agente_Experto", END)

# Compilamos el orquestador
orquestador = grafo.compile()