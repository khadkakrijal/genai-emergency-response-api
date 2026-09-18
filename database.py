import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Missing Supabase URL or key in .env file"
    )


# ---------------------------------------------------------
# Supabase Client
# ---------------------------------------------------------

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


# =========================================================
# INCIDENTS
# =========================================================


def save_incident(
    incident_data: Dict[str, Any]
):
    """
    Save a newly analysed incident.
    """

    try:
        response = (
            supabase
            .table("incidents")
            .insert(incident_data)
            .execute()
        )

        return response.data

    except Exception as e:
        print(
            f"Error saving incident: {e}"
        )
        raise


def get_recent_incidents(
    limit: int = 5
):
    """
    Return the most recently analysed incidents.
    """

    try:
        response = (
            supabase
            .table("incidents")
            .select("*")
            .order(
                "created_at",
                desc=True,
            )
            .limit(limit)
            .execute()
        )

        return response.data

    except Exception as e:
        print(
            f"Error fetching recent incidents: {e}"
        )
        raise


def get_incident_by_id(
    incident_id: str
) -> Optional[Dict[str, Any]]:
    """
    Return one incident using its database ID.
    """

    try:
        response = (
            supabase
            .table("incidents")
            .select("*")
            .eq(
                "id",
                incident_id,
            )
            .maybe_single()
            .execute()
        )

        return response.data

    except Exception as e:
        print(
            f"Error fetching incident "
            f"{incident_id}: {e}"
        )
        raise


def update_incident(
    incident_id: str,
    incident_data: Dict[str, Any],
):
    """
    Update an existing incident.

    This function only updates fields supplied
    in incident_data.
    """

    try:
        response = (
            supabase
            .table("incidents")
            .update(incident_data)
            .eq(
                "id",
                incident_id,
            )
            .execute()
        )

        return response.data

    except Exception as e:
        print(
            f"Error updating incident "
            f"{incident_id}: {e}"
        )
        raise


def delete_incident(
    incident_id: str
):
    """
    Permanently delete an incident.
    """

    try:
        response = (
            supabase
            .table("incidents")
            .delete()
            .eq(
                "id",
                incident_id,
            )
            .execute()
        )

        return response.data

    except Exception as e:
        print(
            f"Error deleting incident "
            f"{incident_id}: {e}"
        )
        raise


# =========================================================
# HISTORICAL INCIDENT DATASET
# =========================================================

def get_historical_incidents(
    limit: int = 5000,
):
    """
    Retrieve a general historical incident sample
    for semantic TF-IDF comparison.
    """

    try:
        response = (
            supabase
            .table("historical_incidents")
            .select(
                "id,"
                "title,"
                "description,"
                "incident_type,"
                "priority_level,"
                "location,"
                "addr,"
                "twp,"
                "timeStamp"
            )
            .order(
                "timeStamp",
                desc=False,
            )
            .limit(limit)
            .execute()
        )

        return response.data or []

    except Exception as e:
        print(
            "Error fetching historical "
            f"incidents: {e}"
        )
        raise


def get_location_historical_incidents(
    location: str,
    limit: int = 1000,
):
    """
    Retrieve additional historical candidates
    using words from the reported location.

    This does NOT calculate the final similarity.
    Final ranking is still performed by the
    TF-IDF/cosine similarity pipeline.
    """

    if not location or not location.strip():
        return []

    try:
        # Break user location into useful components.
        parts = [
            part.strip()
            for part in location.split(",")
            if part.strip()
        ]

        candidates = []

        for part in parts:
            # Skip very short values.
            if len(part) < 3:
                continue

            addr_response = (
                supabase
                .table("historical_incidents")
                .select(
                    "id,"
                    "title,"
                    "description,"
                    "incident_type,"
                    "priority_level,"
                    "location,"
                    "addr,"
                    "twp,"
                    "timeStamp"
                )
                .ilike(
                    "addr",
                    f"%{part}%"
                )
                .limit(limit)
                .execute()
            )

            candidates.extend(
                addr_response.data or []
            )

            twp_response = (
                supabase
                .table("historical_incidents")
                .select(
                    "id,"
                    "title,"
                    "description,"
                    "incident_type,"
                    "priority_level,"
                    "location,"
                    "addr,"
                    "twp,"
                    "timeStamp"
                )
                .ilike(
                    "twp",
                    f"%{part}%"
                )
                .limit(limit)
                .execute()
            )

            candidates.extend(
                twp_response.data or []
            )

        # Remove duplicate records.
        unique = {}

        for incident in candidates:
            incident_id = incident.get("id")

            if incident_id is not None:
                unique[str(incident_id)] = incident

        return list(unique.values())

    except Exception as e:
        print(
            "Error fetching location-aware "
            f"historical incidents: {e}"
        )
        raise