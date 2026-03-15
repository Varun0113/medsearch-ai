from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.pipeline import MedSearchPipeline
from app.core.semantic_pipeline import SemanticSearchPipeline
from app.data_loader import MedicalDataLoader
from app.schemas.api import QueryRequest, QueryResponse, SemanticQueryRequest, SemanticQueryResponse

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="MedSearch AI - Medical Information API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

data_loader = MedicalDataLoader(DATA_DIR)
pipeline = MedSearchPipeline(data_loader)
semantic_pipeline = SemanticSearchPipeline()
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.on_event("startup")
def load_medical_datasets() -> None:
    data_loader.load_datasets()


@app.get("/")
def serve_ui():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_medical_information(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    patient_payload = req.patient.model_dump() if hasattr(req.patient, "model_dump") else req.patient.dict()
    result = pipeline.run(req.query, req.language, patient_payload)
    return QueryResponse(**result)


@app.post("/semantic-query", response_model=SemanticQueryResponse)
def semantic_query(req: SemanticQueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = semantic_pipeline.run(req.query, req.language)
    return SemanticQueryResponse(**result)
