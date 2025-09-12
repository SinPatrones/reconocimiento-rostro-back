import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
import base64
from io import BytesIO
from PIL import Image
import os


class FaceRecognizer:
    def __init__(self, db_path: str = None):
        # Usar variable de entorno o path por defecto
        self.db_path = db_path or os.getenv('DATABASE_PATH', 'face_database.npz')

        # Inicializar soluciones de MediaPipe (usando solo APIs estables)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.database = self.load_database(self.db_path)
        print(f"FaceRecognizer inicializado. Personas en DB: {len(self.database)}")

    def load_database(self, db_path: str) -> Dict[str, np.ndarray]:
        try:
            if os.path.exists(db_path):
                data = np.load(db_path, allow_pickle=True)
                database = dict(zip(data['names'], data['embeddings']))
                print(f"Base de datos cargada desde {db_path} con {len(database)} personas.")
                return database
            else:
                print(f"No se encontró base de datos en {db_path}. Creando nueva.")
                return {}
        except Exception as e:
            print(f"Error cargando base de datos: {e}")
            return {}

    def save_database(self, db_path: str = None):
        db_path = db_path or self.db_path
        if self.database:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            names = list(self.database.keys())
            embeddings = np.array([self.database[name] for name in names])
            np.savez(db_path, names=names, embeddings=embeddings)
            print(f"Base de datos guardada en {db_path} con {len(self.database)} personas.")
            return True
        return False

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def base64_to_image(self, base64_string: str) -> np.ndarray:
        try:
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {e}")

    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extraer embedding usando Face Mesh (API estable)"""
        try:
            with self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
            ) as face_mesh:

                # Convertir a RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    # Extraer todos los landmarks y aplanarlos
                    landmarks = results.multi_face_landmarks[0]
                    embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

                    # Normalizar el embedding
                    embedding = (embedding - np.mean(embedding)) / np.std(embedding)
                    return embedding

        except Exception as e:
            print(f"Error extrayendo embedding: {e}")

        return None

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detectar rostros usando Face Detection (API estable)"""
        faces = []

        try:
            with self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 0=short-range, 1=full-range
                    min_detection_confidence=0.5
            ) as detector:

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = detector.process(image_rgb)

                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w = image.shape[:2]

                        faces.append({
                            'x': int(bbox.xmin * w),
                            'y': int(bbox.ymin * h),
                            'width': int(bbox.width * w),
                            'height': int(bbox.height * h),
                            'confidence': float(detection.score[0])
                        })

        except Exception as e:
            print(f"Error detectando rostros: {e}")

        return faces

    def recognize_face(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        """Comparar embedding con la base de datos"""
        if embedding is None:
            return "Desconocido", 0.0

        best_similarity = -1
        best_person = "Desconocido"

        for person_name, db_embedding in self.database.items():
            try:
                # Asegurarse de que los embeddings tengan la misma dimensión
                min_len = min(len(embedding), len(db_embedding))
                if min_len == 0:
                    continue

                similarity = self.cosine_similarity(
                    embedding[:min_len],
                    db_embedding[:min_len]
                )

                if similarity > best_similarity and similarity > threshold:
                    best_similarity = similarity
                    best_person = person_name
            except Exception as e:
                print(f"Error calculando similitud para {person_name}: {e}")
                continue

        return best_person, best_similarity

    def process_image(self, base64_image: str, threshold: float = 0.6) -> dict:
        """Procesar imagen completa: detección + reconocimiento"""
        try:
            # Convertir base64 a imagen
            image = self.base64_to_image(base64_image)

            if image is None:
                return {
                    "success": False,
                    "error": "No se pudo decodificar la imagen",
                    "faces_detected": 0,
                    "recognitions": []
                }

            # Detectar rostros
            faces = self.detect_faces(image)

            if not faces:
                return {
                    "success": True,
                    "faces_detected": 0,
                    "recognitions": []
                }

            recognitions = []

            for face in faces:
                # Extraer región de la cara
                x, y, w, h = face['x'], face['y'], face['width'], face['height']

                # Asegurarse de que las coordenadas estén dentro de la imagen
                x = max(0, x)
                y = max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)

                if w > 0 and h > 0:
                    face_roi = image[y:y + h, x:x + w]

                    if face_roi.size > 0:
                        # Redimensionar si es muy pequeña para mejor procesamiento
                        if face_roi.shape[0] < 100 or face_roi.shape[1] < 100:
                            face_roi = cv2.resize(face_roi, (150, 150))

                        embedding = self.extract_face_embedding(face_roi)
                        if embedding is not None:
                            name, confidence = self.recognize_face(embedding, threshold)
                            recognitions.append({
                                "name": name,
                                "confidence": float(confidence),
                                "bounding_box": {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h
                                }
                            })
                        else:
                            recognitions.append({
                                "name": "No se pudo extraer características",
                                "confidence": 0.0,
                                "bounding_box": {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h
                                }
                            })
                else:
                    recognitions.append({
                        "name": "Rostro fuera de límites",
                        "confidence": 0.0,
                        "bounding_box": face
                    })

            return {
                "success": True,
                "faces_detected": len(faces),
                "recognitions": recognitions
            }

        except Exception as e:
            print(f"Error en process_image: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces_detected": 0,
                "recognitions": []
            }


# Instancia global del reconocedor
face_recognizer = FaceRecognizer()