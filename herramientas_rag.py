import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone
from openai import OpenAI
from langchain_core.tools import tool

# 1. Cargar variables de entorno
load_dotenv()

# 2. Inicializar Clientes (Fuera de la función para no abrir conexiones en cada consulta)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🌲 Conectando a Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("salary-rag-index")

print("🍃 Conectando a MongoDB...")
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["mercado_laboral_db"]
coleccion_exacta = db["registros_completos"]

@tool
def consultar_estudios_mercado(consulta: str) -> str:
    """
    Busca en los documentos oficiales y estudios de mercado (myDNA, LHH)
    información sobre salarios, perfiles y empresas.
    Úsala siempre que el usuario pregunte por datos de mercado, estudios o rangos salariales documentados.
    """
    try:
        print(f"\n🔍 [RAG LOG] LLM preguntó: '{consulta}'")

        # 1. Embed de la pregunta
        respuesta_embed = openai_client.embeddings.create(
            input=consulta, model="text-embedding-3-small"
        )
        vector_consulta = respuesta_embed.data[0].embedding

        # 2. Query en Pinecone
        resultados_pinecone = pinecone_index.query(
            vector=vector_consulta, top_k=5, include_metadata=True
        )
        print(f"🌲 [RAG LOG] Pinecone encontró {len(resultados_pinecone.matches)} fragmentos.")

        if not resultados_pinecone.matches:
            return "No se encontró información en los estudios de mercado."

        # 3. Extraer IDs base (quitando el '-chunk' si lo tiene)
        ids_base = []
        for match in resultados_pinecone.matches:
            ids_base.append(match.id.replace("-chunk", ""))

        # 4. Consultar en MongoDB
        documentos_mongo = list(coleccion_exacta.find({"id": {"$in": ids_base}}))

        # Mapear los documentos encontrados para fácil acceso
        mongo_dict = {doc["id"]: doc for doc in documentos_mongo}
        print(f"🍃 [RAG LOG] MongoDB hizo match con {len(mongo_dict)} registros exactos.")

        # 5. Armar el Contexto Final
        contexto_final = "📋 DATOS EXTRAÍDOS DE LOS ESTUDIOS DE MERCADO:\n\n"

        for match in resultados_pinecone.matches:
            id_limpio = match.id.replace("-chunk", "")

            # Si el ID está en Mongo, usamos tu hermoso "texto_rag"
            if id_limpio in mongo_dict:
                doc = mongo_dict[id_limpio]
                texto_precalculado = doc.get("texto_rag", "")
                contexto_final += f"✅ {texto_precalculado}\n\n"

            # Si NO está en Mongo (Ej: Es el chunk de la tabla completa), usamos el texto del metadata
            else:
                texto_chunk = match.metadata.get("text", "")
                contexto_final += f"📊 Fragmento/Tabla:\n{texto_chunk}\n\n"

        print(f"✅ [RAG LOG] Contexto final enviado al LLM ({len(contexto_final)} caracteres).")
        return contexto_final

    except Exception as e:
        print(f"❌ [RAG LOG] ERROR INTERNO: {e}")
        return f"Error interno al consultar la base documental: {str(e)}"