# \---

# title: Sueldo

# emoji: 💼

# colorFrom: blue

# colorTo: purple

# sdk: docker

# app\_port: 7860

# pinned: false

# short\_description: Calculadora salarial con XGBoost, RAG agéntico y benchmarks de mercado para Colombia.

# tags:

# &#x20; - salary

# &#x20; - colombia

# &#x20; - xgboost

# &#x20; - rag

# &#x20; - pinecone

# &#x20; - mongodb

# &#x20; - langgraph

# \---

# 

# \# 📊 Calculadora de Remuneración Profesional - IA Dual

# 

# Un sistema end-to-end impulsado por Inteligencia Artificial...

# 

# 

# 

# 📊 Calculadora de Remuneración Profesional - IA Dual

Un sistema end-to-end impulsado por Inteligencia Artificial diseñado para estimar la compensación salarial en el mercado laboral colombiano. El proyecto implementa una arquitectura de **IA Dual**, cruzando predicciones estadísticas de Machine Learning (XGBoost) con datos ejecutivos extraídos de estudios oficiales de mercado a través de un pipeline RAG Agentico.

## 🚀 Descripción y Alcance

El objetivo principal es reducir la asimetría de información salarial para profesionales en Colombia. En lugar de requerir que el usuario llene formularios extensos, el sistema automatiza la entrada de datos leyendo directamente su perfil exportado de LinkedIn (PDF).

El sistema entrega dos perspectivas reconciliadas:

1. **Modelo Estadístico General:** Predicción basada en encuestas nacionales (GEIH) evaluando variables como edad, educación, experiencia y sector.
2. **Benchmark Ejecutivo:** Análisis contextual de estudios de mercado premium (LHH, MyDNA) para afinar rangos en cargos gerenciales o especializados.

## 🏗️ Arquitectura del Sistema

El flujo de la aplicación está dividido en capas independientes y orquestado mediante grafos de estado:

1. **Extracción Inteligente (ETL \& Validation):** Lectura de PDFs e inferencia de variables (edad, sector CIIU) usando GPT-4o-mini con validación de entrada temprana (Early Input Validation) para bloquear documentos irrelevantes.
2. **Inferencia de Machine Learning:** API RESTful (FastAPI) que sirve un modelo XGBoost en milisegundos.
3. **Orquestación Agentica (LangGraph):** Un enrutador (Router) actúa como guardia de seguridad, bloqueando *prompt injections* antes de despertar al Agente ReAct principal.
4. **RAG Híbrido Estructurado:** Búsqueda vectorial semántica (Pinecone) en clústeres de tablas documentales, unida a una consulta exacta de metadatos pre-computados (MongoDB) para eliminar alucinaciones.
5. **Capa de Consenso (LLM Judge):** El Agente evalúa la brecha entre la predicción matemática y el benchmark de mercado, emitiendo alertas de reconciliación automáticamente.

## 🛠️ Stack Tecnológico

* **Frontend:** Gradio, HTML/JS/CSS nativo.
* **Backend \& API:** FastAPI, Uvicorn, Python 3.11.
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas.
* **GenAI \& Agentes:** OpenAI API (GPT-4o-mini, text-embedding-3-small), LangChain, LangGraph.
* **Bases de Datos:** Pinecone (Vectorial Serverless), MongoDB (Documental Local).
* **MLOps \& CI/CD:** MLflow (Tracking local estructurado), GitHub Actions (Evaluación Batch automatizada).

## 📈 Rendimiento y Métricas

### Músculo Predictivo (XGBoost)

* **Variables Entrenadas (Features):** \~150 (incluyendo codificación *one-hot* de códigos CIIU).
* **Métrica de Producción (RMSE):** $625400
* **Monitoreo:** El RMSE es re-evaluado y versionado anualmente mediante triggers de GitHub Actions, guardando el histórico de parámetros en MLflow sin costos de servidor inactivo.

### Cerebro RAG Agentico

* **Estrategia de Chunking:** Generación semántica pre-calculada y almacenamiento tabular híbrido.
* **Recuperación (Retrieval):** Top-K = 20 con expansión de query dinámica (inyección estricta de palabras clave para sectores tecnológicos).
* **Latencia:** Resolución híbrida en Pinecone + MongoDB con un pase de contexto comprimido (<5,000 caracteres) para respuestas casi en tiempo real.

\---

*Desarrollado por Ing. Mario Varon M.Sc - AI Engineer.* ma.varon13@gmail.com

