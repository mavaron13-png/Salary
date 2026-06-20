---
title: Sueldo
emoji: 💼
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 📊 Calculadora de Remuneración Profesional - IA Dual

Un sistema end-to-end impulsado por Inteligencia Artificial diseñado para estimar la compensación salarial en el mercado laboral colombiano.

El proyecto implementa una arquitectura de IA Dual, cruzando predicciones estadísticas de Machine Learning con XGBoost y benchmarks ejecutivos de mercado recuperados mediante RAG agéntico.

## 🚀 Descripción

La aplicación permite subir un perfil profesional exportado desde LinkedIn en PDF. Luego extrae variables profesionales con IA y genera dos perspectivas:

1. **Modelo estadístico de empleabilidad general:** predicción salarial usando XGBoost.
2. **Benchmark ejecutivo de mercado:** recuperación semántica en Pinecone y consulta exacta en MongoDB sobre estudios salariales estructurados.

## 🏗️ Arquitectura

- Extracción de datos desde PDF.
- Inferencia de salario con XGBoost.
- RAG semántico en Pinecone.
- Recuperación exacta de registros salariales desde MongoDB.
- Reconciliación entre estimación estadística y benchmark de mercado.

## 🛠️ Stack

- FastAPI
- Gradio
- XGBoost
- Scikit-Learn
- OpenAI API
- LangChain
- LangGraph
- Pinecone
- MongoDB
- Docker

## 📌 Uso

El Space corre como aplicación Docker en el puerto 7860.

---

Desarrollado por Ing. Mario Varon M.Sc.
