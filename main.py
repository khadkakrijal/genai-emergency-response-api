import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import save_incident, get_recent_incidents
from ai.classifier import EmergencyClassifier

app = FastAPI(
    title="GenAI for Effective Emergency Response API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = EmergencyClassifier()


class IncidentRequest(BaseModel):
    description: str
    location: str | None = None
    incident_time: str | None = None
    people_involved: str | None = None
    weapon_involved: str | None = None
    injury_reported: str | None = None
    location_type: str | None = None


@app.get("/")
def root():
    return {"message": "GenAI Emergency Response API is running!"}


@app.get("/incidents")
def recent_incidents():
    return get_recent_incidents()


@app.post("/analyse-incident")
def analyse_incident(data: IncidentRequest):
    start_time = time.time()

    incident_data = data.model_dump()

    ai_result = classifier.classify(incident_data)

    processing_time_ms = int((time.time() - start_time) * 1000)

    incident_record = {
        "description": data.description,
        "location": data.location,
        "incident_time": data.incident_time,
        "people_involved": data.people_involved,
        "weapon_involved": data.weapon_involved,
        "injury_reported": data.injury_reported,
        "location_type": data.location_type,

        "incident_type": ai_result.get("incident_type"),
        "risk_level": ai_result.get("risk_level"),
        "priority": ai_result.get("priority"),
        "confidence_score": ai_result.get("confidence_score"),

        "summary": ai_result.get("summary"),
        "recommended_response": ai_result.get("recommended_response"),
        "reasoning": ai_result.get("reasoning"),

        "responders": ai_result.get("responders"),
        "key_risks": ai_result.get("key_risks"),

        "ai_model": "llama3.2",
        "processing_time_ms": processing_time_ms,
        "status": "Completed",
    }

    saved_incident = save_incident(incident_record)

    return {
        **ai_result,
        "processing_time_ms": processing_time_ms,
        "saved_incident": saved_incident,
    }