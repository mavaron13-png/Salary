# 1. Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# 2. Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema operativo requeridas por XGBoost y otros
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar los requerimientos e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO el código de tu proyecto al contenedor
COPY . .

# 6. Exponer el puerto 7860 (El único que Hugging Face permite)
EXPOSE 7860

# 7. El comando maestro para arrancar la API y servir el Frontend
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860"]