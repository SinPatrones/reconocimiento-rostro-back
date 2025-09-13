FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para mejor caching
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorio para la base de datos
RUN mkdir -p /data

# Copiar código de la aplicación
COPY app/ ./app/

# Script de entrada
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Exponer puerto
EXPOSE 8000

# Comando de inicio
ENTRYPOINT ["./entrypoint.sh"]