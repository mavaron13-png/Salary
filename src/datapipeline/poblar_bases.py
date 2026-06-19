import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# 1. Cargar variables de entorno
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Configuración de MongoDB
print("🍃 Conectando a MongoDB...")
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["mercado_laboral_db"]
coleccion_exacta = db["registros_completos"]

# 3. Configuración de Pinecone
print("🌲 Conectando a Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "salary-rag-index"

# Crear el índice en Pinecone si no existe (Dimensión 1536 para OpenAI text-embedding-3-small)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # Ajusta tu región si es distinta
    )
pinecone_index = pc.Index(index_name)

# --- FUNCIONES DE CARGA ---

def cargar_a_mongo(ruta_archivo):
    """Sube registros JSONL directamente a MongoDB"""
    print(f"🔄 Procesando {ruta_archivo} para MongoDB...")
    registros = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            if linea.strip():
                registros.append(json.loads(linea))
    
    if registros:
        coleccion_exacta.insert_many(registros)
        print(f"✅ {len(registros)} registros subidos a MongoDB desde {ruta_archivo}")

def cargar_a_pinecone(ruta_archivo):
    """Vectoriza y sube chunks a Pinecone por lotes (batches)"""
    print(f"🔄 Procesando {ruta_archivo} para Pinecone...")
    lote_vectores = []
    batch_size = 100 # Pinecone prefiere cargas en lotes pequeños
    
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    for i, linea in enumerate(lineas):
        if not linea.strip(): continue
        
        datos = json.loads(linea)
        id_chunk = datos.get("id", f"chunk_{i}")
        texto_chunk = datos.get("text", "")
        metadata = datos.get("metadata", {})
        
        # 1. Generar el Embedding con OpenAI
        respuesta_embed = openai_client.embeddings.create(
            input=texto_chunk,
            model="text-embedding-3-small"
        )
        vector = respuesta_embed.data[0].embedding
        
        # 2. Asegurar que el texto viaje en la metadata para el Agente
        metadata["text"] = texto_chunk 
        
        lote_vectores.append({"id": id_chunk, "values": vector, "metadata": metadata})
        
        # Subir cuando el lote esté lleno o sea el final
        if len(lote_vectores) >= batch_size or i == len(lineas) - 1:
            pinecone_index.upsert(vectors=lote_vectores)
            print(f"⬆️ Lote de {len(lote_vectores)} vectores subido...")
            lote_vectores = []
            
    print(f"✅ Todos los chunks de {ruta_archivo} están en Pinecone.")

# --- EJECUCIÓN DEL PIPELINE ---

if __name__ == "__main__":
    # 1. Subir los datos crudos/exactos a Mongo
    cargar_a_mongo(r"C:\Users\ma_va\Documents\Salary\RAG\mydna_colombia_records.jsonl")
    cargar_a_mongo(r"C:\Users\ma_va\Documents\Salary\RAG\estudio_laboral_lhh_records.jsonl")

    # 2. Subir los fragmentos vectorizados a Pinecone
    cargar_a_pinecone(r"C:\Users\ma_va\Documents\Salary\RAG\estudio_laboral_lhh_chunks.jsonl")
    cargar_a_pinecone(r"C:\Users\ma_va\Documents\Salary\RAG\mydna_colombia_chunks.jsonl")
    
    print("🚀 ¡Ingesta de datos finalizada con éxito! Las bases están listas para el Agente.")