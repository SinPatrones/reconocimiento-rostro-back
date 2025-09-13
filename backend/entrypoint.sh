#!/bin/bash

# Crear directorio de datos si no existe
mkdir -p /data

# Crear enlace simbólico si la base de datos existe en el volumen
if [ -f "/data/face_database.npz" ]; then
    echo "Enlazando base de datos existente..."
    ln -sf /data/face_database.npz /app/face_database.npz
else
    echo "No se encontró base de datos existente. Se creará una nueva."
    # Crear archivo vacío para el enlace
    touch /data/face_database.npz
    ln -sf /data/face_database.npz /app/face_database.npz
fi

# Verificar que las dependencias estén instaladas
echo "Verificando dependencias..."
python -c "
try:
    import mediapipe
    print(f'✓ MediaPipe {mediapipe.__version__} instalado')
except ImportError as e:
    print(f'✗ Error importando MediaPipe: {e}')
    exit(1)

try:
    import cv2
    print(f'✓ OpenCV {cv2.__version__} instalado')
except ImportError as e:
    print(f'✗ Error importando OpenCV: {e}')
    exit(1)

try:
    import fastapi
    print(f'✓ FastAPI instalado')
except ImportError as e:
    print(f'✗ Error importando FastAPI: {e}')
    exit(1)

print('Todas las dependencias verificadas correctamente')
"

# Ejecutar la aplicación
echo "Iniciando Face Recognition API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload