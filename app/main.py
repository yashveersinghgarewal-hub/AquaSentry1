"""
AquaSentry backend API.

Run locally with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for interactive API docs (Swagger UI).

Flow:
    Arduino (sensor + HC-05 Bluetooth) --serial--> Bridge script (laptop/RPi)
        --HTTP POST--> /api/readings  --> classified & stored in DB
    Website dashboard --HTTP GET--> /api/readings, /api/stats, etc.
"""

from datetime import datetime, timedelta
import os
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session

from . import models, schemas
from .database import engine, get_db, Base, ensure_sqlite_columns
from .classification import classify_arsenic

# Create tables on startup if they don't exist yet.
Base.metadata.create_all(bind=engine)
ensure_sqlite_columns()

app = FastAPI(
    title="AquaSentry API",
    description="Backend for the AI-assisted arsenic-detection water robot.",
    version="1.0.0",
)

# Allow the local dashboard and simple static deployments to call the API.
# Set AQUASENTRY_ALLOWED_ORIGINS to a comma-separated list in production.
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "AQUASENTRY_ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8001,http://127.0.0.1:8001",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Meta"])
def health_check():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_or_create_device(db: Session, device_code: str) -> models.Device:
    device = db.query(models.Device).filter_by(device_code=device_code).first()
    now = datetime.utcnow()
    if device is None:
        device = models.Device(device_code=device_code, first_seen=now, last_seen=now)
        db.add(device)
        db.commit()
        db.refresh(device)
    else:
        device.last_seen = now
        db.commit()
    return device


# ---------------------------------------------------------------------------
# Readings: ingest (called by the bridge/Arduino side)
# ---------------------------------------------------------------------------

@app.post("/api/readings", response_model=schemas.ReadingResponse, tags=["Readings"])
def create_reading(payload: schemas.ReadingCreate, db: Session = Depends(get_db)):
    """
    Ingest a new arsenic reading from the robot.
    Classifies the arsenic level and stores everything in the database.
    """
    try:
        result = classify_arsenic(payload.arsenic_ppb)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    device = _get_or_create_device(db, payload.device_code)

    reading = models.Reading(
        device_id=device.id,
        arsenic_ppb=payload.arsenic_ppb,
        classification=result.label,
        ph=payload.ph,
        temperature_c=payload.temperature_c,
        conductivity_us_cm=payload.conductivity_us_cm,
        turbidity_ntu=payload.turbidity_ntu,
        dissolved_oxygen_mg_l=payload.dissolved_oxygen_mg_l,
        latitude=payload.latitude,
        longitude=payload.longitude,
        confidence=payload.confidence,
        battery_pct=payload.battery_pct,
        source_label=payload.source_label,
        recorded_at=payload.recorded_at or datetime.utcnow(),
        received_at=datetime.utcnow(),
        raw_payload=payload.raw_payload,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    return schemas.ReadingResponse(
        reading=schemas.ReadingOut.model_validate(reading),
        label=result.label,
        severity=result.severity,
        message=result.message,
        action=result.action,
    )


# ---------------------------------------------------------------------------
# Readings: query (called by the website dashboard)
# ---------------------------------------------------------------------------

@app.get("/api/readings", response_model=List[schemas.ReadingOut], tags=["Readings"])
def list_readings(
    device_code: Optional[str] = Query(None, description="Filter by device code"),
    classification: Optional[str] = Query(
        None, description="Filter by Safe / Caution / Unsafe / Hazardous"
    ),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(models.Reading)
    if device_code:
        q = q.join(models.Device).filter(models.Device.device_code == device_code)
    if classification:
        q = q.filter(models.Reading.classification == classification)
    q = q.order_by(models.Reading.recorded_at.desc()).offset(offset).limit(limit)
    return q.all()


@app.get("/api/readings/latest", response_model=schemas.ReadingOut, tags=["Readings"])
def latest_reading(
    device_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(models.Reading)
    if device_code:
        q = q.join(models.Device).filter(models.Device.device_code == device_code)
    reading = q.order_by(models.Reading.recorded_at.desc()).first()
    if reading is None:
        raise HTTPException(status_code=404, detail="No readings found yet.")
    return reading


@app.get("/api/readings/{reading_id}", response_model=schemas.ReadingOut, tags=["Readings"])
def get_reading(reading_id: int, db: Session = Depends(get_db)):
    reading = db.query(models.Reading).filter_by(id=reading_id).first()
    if reading is None:
        raise HTTPException(status_code=404, detail="Reading not found.")
    return reading


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

@app.get("/api/devices", response_model=List[schemas.DeviceOut], tags=["Devices"])
def list_devices(db: Session = Depends(get_db)):
    return db.query(models.Device).order_by(models.Device.last_seen.desc()).all()


# ---------------------------------------------------------------------------
# Stats (for dashboard summary cards / charts)
# ---------------------------------------------------------------------------

@app.get("/api/stats", response_model=schemas.StatsOut, tags=["Meta"])
def get_stats(device_code: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(models.Reading)
    if device_code:
        q = q.join(models.Device).filter(models.Device.device_code == device_code)

    total = q.count()
    latest = q.order_by(models.Reading.recorded_at.desc()).first()
    avg = db.query(func.avg(models.Reading.arsenic_ppb))
    if device_code:
        avg = avg.join(models.Device).filter(models.Device.device_code == device_code)
    avg_value = avg.scalar()

    breakdown = {"Safe": 0, "Caution": 0, "Unsafe": 0, "Hazardous": 0}
    rows = q.with_entities(
        models.Reading.classification, func.count(models.Reading.id)
    ).group_by(models.Reading.classification).all()
    for label, count in rows:
        breakdown[label] = count

    devices_count = db.query(models.Device).count()
    if device_code:
        devices_count = db.query(models.Device).filter(
            models.Device.device_code == device_code
        ).count()

    return schemas.StatsOut(
        total_readings=total,
        devices_count=devices_count,
        latest_reading=latest,
        breakdown=breakdown,
        average_arsenic_ppb=round(avg_value, 3) if avg_value is not None else None,
    )


# ---------------------------------------------------------------------------
# Dashboard projection
# ---------------------------------------------------------------------------

SAFE_LIMIT_MG_L = 0.010


@app.get("/api/dashboard/latest", response_model=schemas.DashboardReading, tags=["Dashboard"])
def dashboard_latest(device_code: Optional[str] = Query(None), db: Session = Depends(get_db)):
    q = db.query(models.Reading)
    if device_code:
        q = q.join(models.Device).filter(models.Device.device_code == device_code)
    reading = q.order_by(models.Reading.recorded_at.desc()).first()
    if reading is None:
        raise HTTPException(status_code=404, detail="No readings found yet.")

    confirmed_count = 0
    if reading.classification in ("Unsafe", "Hazardous"):
        cutoff = datetime.utcnow() - timedelta(hours=24)
        confirmed_count = (
            db.query(models.Reading.device_id)
            .filter(
                models.Reading.classification.in_(["Unsafe", "Hazardous"]),
                models.Reading.recorded_at >= cutoff,
                models.Reading.device_id != reading.device_id,
            )
            .distinct()
            .count()
        )

    patrol_minutes = max(
        0,
        round((datetime.utcnow() - reading.recorded_at).total_seconds() / 60, 1),
    )
    return schemas.DashboardReading(
        contaminant="Arsenic (As)",
        value=round(reading.arsenic_ppb / 1000, 4),
        unit="mg/L",
        safeLimit=SAFE_LIMIT_MG_L,
        confidencePct=round(reading.confidence * 100, 1) if reading.confidence is not None else None,
        confirmedByHomes=confirmed_count,
        source=reading.source_label or "Groundwater",
        detectedAt=reading.recorded_at.strftime("%H:%M"),
        batteryPct=reading.battery_pct,
        lastPatrolMin=patrol_minutes,
        nextPatrolMin=None,
        classification=reading.classification,
    )
