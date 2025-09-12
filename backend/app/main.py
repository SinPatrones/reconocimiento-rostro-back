from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import os
from .utils.face_utils import face_recognizer

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Face Recognition API", "database_path": face_recognizer.db_path}


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "database_path": face_recognizer.db_path,
        "database_size": len(face_recognizer.database),
        "known_people": list(face_recognizer.database.keys())
    }


@app.post("/api/recognize")
async def recognize_face(
        image: str = Form(..., description="Base64 encoded image"),
        threshold: float = Form(0.75, description="Similarity threshold")
):
    try:
        result = face_recognizer.process_image(image, threshold)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_person(
        name: str = Form(..., description="Name of the person"),
        images: List[str] = Form(..., description="List of base64 encoded images")
):
    try:
        embeddings = []

        for base64_img in images:
            image = face_recognizer.base64_to_image(base64_img)
            embedding = face_recognizer.extract_face_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            face_recognizer.database[name] = avg_embedding
            face_recognizer.save_database()

            return {
                "success": True,
                "message": f"Persona '{name}' entrenada con {len(embeddings)} imágenes",
                "database_size": len(face_recognizer.database),
                "database_path": face_recognizer.db_path
            }
        else:
            raise HTTPException(status_code=400, detail="No se pudieron extraer embeddings de las imágenes")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/people")
async def get_known_people():
    return {
        "people": list(face_recognizer.database.keys()),
        "count": len(face_recognizer.database),
        "database_path": face_recognizer.db_path
    }


@app.delete("/api/people/{name}")
async def delete_person(name: str):
    if name in face_recognizer.database:
        del face_recognizer.database[name]
        face_recognizer.save_database()
        return {"success": True, "message": f"Persona '{name}' eliminada"}
    else:
        raise HTTPException(status_code=404, detail="Persona no encontrada")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)