FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Limpieza defensiva contra el paquete viejo de Pinecone
RUN pip uninstall -y pinecone-client pinecone || true

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Reinstalar Pinecone correcto al final, por si alguna dependencia mete basura vieja
RUN pip install --no-cache-dir --force-reinstall "pinecone>=5.0.0"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "7860", "--no-server-header"]