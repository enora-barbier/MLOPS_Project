"""
FastAPI application for the Paris 15e property valuation & anomaly detection tool.

Endpoints:
    GET /score   — score a property listing
    GET /health  — model metadata and service status
    GET /        — serves the web UI (static/index.html)

Usage:
    python -m uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from score import load_model, score_listing
from validate import ScoreRequest

app = FastAPI(
    title="Paris 15e Property Valuation",
    description="Estimate fair market value and flag pricing anomalies for properties in the 15th arrondissement of Paris.",
)

model = load_model()


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.get("/score")
def score(
    surface: float = Query(..., gt=0, le=1000, description="Living area in m²"),
    nb_room: int = Query(..., ge=0, le=50, description="Number of main rooms"),
    property_type: str = Query(..., description="'Appartement' or 'Maison'"),
    section_id: str = Query(..., min_length=10, max_length=12, description="Cadastral section code, e.g. '75115000CG'"),
    price: float = Query(..., gt=0, description="Listed price in EUR"),
):
    req = ScoreRequest(
        surface=surface,
        nb_room=nb_room,
        property_type=property_type,
        section_id=section_id,
        price=price,
    )

    return score_listing(
        model,
        surface=req.surface,
        nb_room=req.nb_room,
        is_appartement=req.is_appartement,
        parcel_id=req.parcel_id,
        price=req.price,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": {
            "n_train": model.get("n_train"),
            "r_squared": model.get("r_squared"),
            "sigma": model.get("sigma"),
            "mae_eur": model.get("mae_eur"),
            "trained_at": model.get("trained_at"),
            "n_sections": len(model.get("section_means", {})),
        },
    }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
