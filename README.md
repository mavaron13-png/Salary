# 🇨🇴 Salary - Asistente de Remuneración Profesional en Colombia

## 📌 Descripción del Proyecto
[cite_start]Este proyecto es un servicio web interactivo diseñado para que los usuarios puedan conocer una remuneración adecuada según su perfil profesional en el mercado laboral de Colombia[cite: 2]. [cite_start]A través de un chatbot inteligente, los usuarios pueden interactuar mediante texto, enviar audios o subir su Currículum Vitae en formato PDF [cite: 2] para recibir un análisis de sueldo, comparar su perfil con estudios del mercado y buscar ofertas reales.

## 🏗️ Arquitectura y Stack Tecnológico

El sistema está construido bajo una arquitectura modular orientada a la baja latencia y optimización de costos en la nube:

### 1. Interfaz de Usuario (Capa de Presentación)
* [cite_start]**Gradio:** Frontend temporal (PoC) que maneja la interfaz de chat, la grabación de audios y la carga de documentos PDF[cite: 63].

### 2. Backend (Capa de API)
* [cite_start]**FastAPI:** El motor principal que recibe las peticiones, coordina los servicios y devuelve la respuesta al usuario[cite: 64].
* [cite_start]**PyMuPDF:** Utilizado localmente para la extracción rápida de texto desde los PDFs (CVs)[cite: 66].
* [cite_start]**Amazon Transcribe:** Servicio de AWS utilizado para convertir el audio enviado por el usuario a texto (Speech-to-Text)[cite: 67].

### 3. Orquestación y Cerebro (RAG Agéntico)
* [cite_start]**LLM:** Amazon Bedrock (Claude 3 Haiku) para procesar el lenguaje natural con baja latencia y bajo costo[cite: 68].
* [cite_start]**LangGraph:** Orquestador de flujos determinísticos que incluye un *router* para desviar temas que no sean laborales con respuestas estructuradas[cite: 69].

### 4. Sistema Multi-Agente (Tools)
* [cite_start]**RAG Documental:** Base de datos vectorial (ChromaDB/FAISS) para buscar perfiles y datos en documentos en PDF sobre estudios del mercado laboral en Colombia[cite: 5, 70].
* [cite_start]**Mercado Actual:** Integración con SerpAPI para buscar ofertas de empleo similares en tiempo real y devolverlas en formato JSON[cite: 71].
* [cite_start]**Predictor de Sueldo:** Consulta a un modelo de Machine Learning servido localmente[cite: 72].

### 5. MLOps y Machine Learning
* [cite_start]**XGBoost:** Modelo basado en árboles de decisión (Gradient Boosting) para realizar la predicción tabular del sueldo basada en años de experiencia, educación y sector[cite: 53, 56].
* [cite_start]**MLflow:** Plataforma utilizada para entrenar el modelo, registrar métricas (RMSE, MAE), realizar *tuning* de hiperparámetros y versionar el modelo predictivo[cite: 62, 73].