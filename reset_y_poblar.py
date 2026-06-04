import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pinecone import Pinecone
from openai import OpenAI

# 1. Cargar variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Conexiones
print("🍃 Conectando a MongoDB...")
mongo_client = MongoClient(os.getenv("MONGO_URI"))
coleccion_exacta = mongo_client["mercado_laboral_db"]["registros_completos"]

print("🌲 Conectando a Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "salary-rag-index"
pinecone_index = pc.Index(index_name)

def limpiar_bases():
    print("🧹 Vaciando colección en MongoDB...")
    coleccion_exacta.delete_many({})
    
    print("🧹 Vaciando vectores en Pinecone...")
    try:
        pinecone_index.delete(delete_all=True) # Borra todos los vectores del índice
    except Exception as e:
        print(f"⚠️ Nota Pinecone (puede ser normal si estaba vacío): {e}")

def poblar_mongo(ruta_records):
    print(f"\n📥 Inyectando {ruta_records} en MongoDB...")
    registros = []
    with open(ruta_records, 'r', encoding='utf-8') as f:
        for linea in f:
            if linea.strip():
                registros.append(json.loads(linea))
    
    if registros:
        coleccion_exacta.insert_many(registros)
        print(f"✅ {len(registros)} registros inyectados en MongoDB.")

def poblar_pinecone(ruta_chunks):
    print(f"\n📥 Vectorizando e inyectando {ruta_chunks} en Pinecone...")
    lote_vectores = []
    batch_size = 100 
    total_subidos = 0
    
    with open(ruta_chunks, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    for i, linea in enumerate(lineas):
        if not linea.strip(): continue
        chunk = json.loads(linea)
        
        texto = chunk.get("text", "")
        # Embed del texto
        embed_response = openai_client.embeddings.create(
            input=texto, model="text-embedding-3-small"
        )
        vector = embed_response.data[0].embedding
        
        # Preparamos la metadata
        metadata = {"text": texto}
        if "tipo_chunk" in chunk:
            metadata["tipo_chunk"] = chunk["tipo_chunk"]
            
        lote_vectores.append({"id": chunk["id"], "values": vector, "metadata": metadata})
        
        if len(lote_vectores) >= batch_size or i == len(lineas) - 1:
            pinecone_index.upsert(vectors=lote_vectores)
            total_subidos += len(lote_vectores)
            print(f"⬆️ Lote subido... ({total_subidos} vectores en total)")
            lote_vectores = []

    print("✅ Población de Pinecone terminada.")

if __name__ == "__main__":
    limpiar_bases()
    poblar_mongo(r"C:\Users\ma_va\Documents\Salary\RAG\salary_records_normalized_merged.jsonl")
    poblar_pinecone(r"C:\Users\ma_va\Documents\Salary\RAG\salary_chunks_from_normalized_records.jsonl")
    print("\n🚀 ¡Bases de datos reiniciadas y listas con la nueva estructura normalizada!")