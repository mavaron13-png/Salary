import os
import json          # 🔒 FALTABA: sin esto, el primer chunk tipo "tabla" hacía NameError → crash
import logging
import certify
from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone
from openai import OpenAI
from langchain_core.tools import tool



load_dotenv()
logger = logging.getLogger("sueldo.rag")

# 2. Clientes inicializados una vez (fuera de la función)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger.info("Conectando a Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("salary-rag-index")

logger.info("Conectando a MongoDB...")
mongo_client = MongoClient(
    os.getenv("MONGO_URI"),
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,   # 🔧 baja de 20s: falla rápido, no cuelga la UI
)
db = mongo_client["mercado_laboral_db"]
coleccion_exacta = db["registros_completos"]


@tool
def consultar_estudios_mercado(consulta: str) -> str:
    """Busca en estudios de mercado (myDNA, LHH) salarios, perfiles y rangos documentados.
    Úsala cuando el usuario pregunte por datos de mercado, estudios o rangos salariales."""
    try:
        # 🔒 logger.debug, no print: en prod (nivel INFO) NO se emite la consulta del usuario
        #    ni el contexto recuperado. Antes los prints filtraban todo a stdout/logs del Space.
        logger.debug("LLM preguntó: %s", consulta)

        # 1. Embed de la pregunta
        emb = openai_client.embeddings.create(input=consulta, model="text-embedding-3-small")
        vector = emb.data[0].embedding

        # 2. Query en Pinecone
        res = pinecone_index.query(vector=vector, top_k=20, include_metadata=True)
        logger.debug("Pinecone: %d fragmentos.", len(res.matches))
        if not res.matches:
            return "No se encontró información en los estudios de mercado."

        # 3. Extraer IDs (tabla o fila)
        all_ids = []
        for match in res.matches:
            meta = match.metadata
            tipo = meta.get("tipo_chunk", "fila")
            if tipo == "tabla" and "record_ids" in meta:
                ids = meta["record_ids"]
                if isinstance(ids, str):
                    try:
                        ids = json.loads(ids)   # 🔒 ahora json sí existe
                    except Exception:
                        ids = [ids]
                if isinstance(ids, list):
                    all_ids.extend(ids)
            else:
                all_ids.append(match.id.replace("-chunk", ""))

        all_ids = list(set(all_ids))

        # 4. MongoDB de un solo golpe. Nota: los IDs salen de TU índice, no del usuario.
        docs = list(coleccion_exacta.find({"id": {"$in": all_ids}}))
        mongo_dict = {d["id"]: d for d in docs}
        logger.debug("MongoDB: %d registros.", len(mongo_dict))

        # 5. Armar contexto
        contexto = "📋 DATOS EXTRAÍDOS DE LOS ESTUDIOS DE MERCADO:\n\n"
        for doc in mongo_dict.values():
            txt = doc.get("texto_rag", "")
            if txt:
                contexto += f"✅ {txt}\n\n"

        logger.debug("Contexto final: %d chars.", len(contexto))
        return contexto

    except Exception as e:
        # 🔒 Detalle al log, mensaje neutro al LLM. Antes devolvías str(e) → leak de
        #    errores de Mongo/Pinecone (fragmentos de URI, hosts, paths) vía el modelo.
        logger.warning("consultar_estudios_mercado falló: %s", e)
        return "No se pudo consultar la base documental en este momento."
