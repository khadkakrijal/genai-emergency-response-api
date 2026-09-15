import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    save_incident,
    get_recent_incidents,
    get_incident_by_id,
    update_incident,
    delete_incident,
)

from ai.classifier import EmergencyClassifier


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="GenAI for Effective Emergency Response API",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://genai-emergency-response-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AI CLASSIFIER
# =========================================================

classifier = EmergencyClassifier()


# =========================================================
# PYDANTIC MODELS
# =========================================================

class IncidentRequest(BaseModel):
    description: str
    location: str | None = None
    incident_time: str | None = None
    people_involved: str | None = None
    weapon_involved: str | None = None
    injury_reported: str | None = None
    location_type: str | None = None


class IncidentUpdateRequest(BaseModel):
    description: str | None = None
    location: str | None = None
    incident_time: str | None = None
    people_involved: str | None = None
    weapon_involved: str | None = None
    injury_reported: str | None = None
    location_type: str | None = None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "GenAI Emergency Response API is running!"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "GenAI Emergency Response API",
        "ai_provider": (
            "Groq"
            if os.getenv(
                "USE_GROQ",
                "false",
            ).lower() == "true"
            else "Ollama"
        ),
    }


# =========================================================
# GET INCIDENT HISTORY
# =========================================================

@app.get("/incidents")
def recent_incidents():
    try:
        return get_recent_incidents()

    except Exception as e:
        print(
            f"Error retrieving incidents: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve incidents",
        )


# =========================================================
# GET SINGLE INCIDENT
# =========================================================

@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    try:
        incident = get_incident_by_id(
            incident_id
        )

        if not incident:
            raise HTTPException(
                status_code=404,
                detail="Incident not found",
            )

        return incident

    except HTTPException:
        raise

    except Exception as e:
        print(
            f"Error retrieving incident "
            f"{incident_id}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve incident",
        )


# =========================================================
# ANALYSE NEW INCIDENT
# =========================================================

@app.post("/analyse-incident")
def analyse_incident(
    data: IncidentRequest
):
    start_time = time.time()

    try:
        # ---------------------------------------------
        # Convert incoming request into dictionary
        # ---------------------------------------------

        incident_data = data.model_dump()

        # ---------------------------------------------
        # Run AI analysis
        # ---------------------------------------------

        ai_result = classifier.classify(
            incident_data
        )

        processing_time_ms = int(
            (time.time() - start_time) * 1000
        )

        # ---------------------------------------------
        # Determine AI provider/model
        # ---------------------------------------------

        using_groq = (
            os.getenv(
                "USE_GROQ",
                "false",
            ).lower()
            == "true"
        )

        ai_model = (
            "Groq"
            if using_groq
            else "Ollama Llama 3.2"
        )

        # ---------------------------------------------
        # Build database record
        # ---------------------------------------------

        incident_record = {
            # Original incident information

            "description":
                data.description,

            "location":
                data.location,

            "incident_time":
                data.incident_time,

            "people_involved":
                data.people_involved,

            "weapon_involved":
                data.weapon_involved,

            "injury_reported":
                data.injury_reported,

            "location_type":
                data.location_type,

            # AI assessment

            "incident_type":
                ai_result.get(
                    "incident_type"
                ),

            "risk_level":
                ai_result.get(
                    "risk_level"
                ),

            "priority":
                ai_result.get(
                    "priority"
                ),

            "confidence_score":
                ai_result.get(
                    "confidence_score"
                ),

            "summary":
                ai_result.get(
                    "summary"
                ),

            "recommended_response":
                ai_result.get(
                    "recommended_response"
                ),

            "reasoning":
                ai_result.get(
                    "reasoning"
                ),

            "responders":
                ai_result.get(
                    "responders"
                ),

            "key_risks":
                ai_result.get(
                    "key_risks"
                ),

            # System information

            "ai_model":
                ai_model,

            "processing_time_ms":
                processing_time_ms,

            "status":
                "Completed",
        }

        # ---------------------------------------------
        # Save incident
        # ---------------------------------------------

        saved_incident = save_incident(
            incident_record
        )

        # ---------------------------------------------
        # Return AI result
        # ---------------------------------------------

        return {
            **ai_result,
            "processing_time_ms":
                processing_time_ms,
            "saved_incident":
                saved_incident,
        }

    except Exception as e:
        print(
            f"Incident analysis error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Incident analysis failed",
        )


# =========================================================
# UPDATE INCIDENT
# =========================================================

@app.put("/incidents/{incident_id}")
def edit_incident(
    incident_id: str,
    data: IncidentUpdateRequest,
):
    try:
        # ---------------------------------------------
        # Check incident exists
        # ---------------------------------------------

        existing_incident = (
            get_incident_by_id(
                incident_id
            )
        )

        if not existing_incident:
            raise HTTPException(
                status_code=404,
                detail="Incident not found",
            )

        # ---------------------------------------------
        # Only include fields actually supplied
        # ---------------------------------------------

        update_data = (
            data.model_dump(
                exclude_unset=True
            )
        )

        if not update_data:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No incident fields "
                    "provided for update"
                ),
            )

        # ---------------------------------------------
        # Update database
        # ---------------------------------------------

        updated = update_incident(
            incident_id,
            update_data,
        )

        if not updated:
            raise HTTPException(
                status_code=404,
                detail="Incident not found",
            )

        return {
            "message":
                "Incident updated successfully",
            "incident":
                updated[0]
                if isinstance(
                    updated,
                    list,
                )
                and updated
                else updated,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            f"Error updating incident "
            f"{incident_id}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to update incident",
        )


# =========================================================
# DELETE INCIDENT
# =========================================================

@app.delete("/incidents/{incident_id}")
def remove_incident(
    incident_id: str
):
    try:
        # ---------------------------------------------
        # Check incident exists
        # ---------------------------------------------

        existing_incident = (
            get_incident_by_id(
                incident_id
            )
        )

        if not existing_incident:
            raise HTTPException(
                status_code=404,
                detail="Incident not found",
            )

        # ---------------------------------------------
        # Delete
        # ---------------------------------------------

        delete_incident(
            incident_id
        )

        return {
            "message":
                "Incident deleted successfully",
            "id":
                incident_id,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            f"Error deleting incident "
            f"{incident_id}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to delete incident",
        )