#!/bin/bash

# Crear enlace simbólico si la base de datos existe en el volumen
if [ -f "/data/face_database.npz" ]; then
    echo "Enlazando base de datos existente..."
    ln -sf /data/face_database.npz /app/face_database.npz
else
    echo "No se encontró base de datos existente. Se creará una nueva."
    touch /data/face_database.npz
    ln -sf /data/face_database.npz /app/face_database.npz
fi

# Ejecutar la aplicación
echo "Iniciando Face Recognition API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload