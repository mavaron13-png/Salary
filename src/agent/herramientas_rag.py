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

        # 2. Query en Pinecone (Subimos top_k a 20 para atrapar todas las variantes de empresa)
        resultados_pinecone = pinecone_index.query(
            vector=vector_consulta, top_k=20, include_metadata=True
        )
        print(f"🌲 [RAG LOG] Pinecone encontró {len(resultados_pinecone.matches)} fragmentos.")

        if not resultados_pinecone.matches:
            return "No se encontró información en los estudios de mercado."

        # 3. Extraer IDs (Detectando dinámicamente si es tabla o fila)
        all_record_ids = []

        for match in resultados_pinecone.matches:
            meta = match.metadata
            tipo_chunk = meta.get("tipo_chunk", "fila")  # Por defecto asumimos fila

            if tipo_chunk == "tabla" and "record_ids" in meta:
                # Extraemos la lista de IDs hijos de la tabla
                ids_tabla = meta["record_ids"]
                # Seguro por si Pinecone guardó la lista como un string serializado
                if isinstance(ids_tabla, str):
                    try:
                        ids_tabla = json.loads(ids_tabla)
                    except:
                        ids_tabla = [ids_tabla]

                if isinstance(ids_tabla, list):
                    all_record_ids.extend(ids_tabla)
            else:
                # Lógica tradicional para chunks tipo fila
                id_base = match.id.replace("-chunk", "")
                all_record_ids.append(id_base)

        # Limpiar duplicados para no sobrecargar a MongoDB
        all_record_ids = list(set(all_record_ids))

        # 4. Consultar en MongoDB todos los registros de un solo golpe
        documentos_mongo = list(coleccion_exacta.find({"id": {"$in": all_record_ids}}))

        mongo_dict = {doc["id"]: doc for doc in documentos_mongo}
        print(f"🍃 [RAG LOG] MongoDB hizo match con {len(mongo_dict)} registros exactos.")

        # 5. Armar el Contexto Final
        contexto_final = "📋 DATOS EXTRAÍDOS DE LOS ESTUDIOS DE MERCADO:\n\n"

        for doc_id, doc in mongo_dict.items():
            # Inyectamos directamente los textos masticados de Mongo
            texto_precalculado = doc.get("texto_rag", "")
            if texto_precalculado:
                contexto_final += f"✅ {texto_precalculado}\n\n"

        print(f"✅ [RAG LOG] Contexto final enviado al LLM ({len(contexto_final)} caracteres).")
        return contexto_final

    except Exception as e:
        print(f"❌ [RAG LOG] ERROR INTERNO: {e}")
        return f"Error interno al consultar la base documental: {str(e)}"