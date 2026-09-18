from database import (
    get_historical_incidents,
    get_location_historical_incidents,
)

from similarity import find_ai_similar_incidents


def retrieve_similar_incidents(
    description: str,
    location: str = "",
    top_k: int = 5,
):
    if not description or not description.strip():
        return []

    # -------------------------------------------------
    # General historical corpus
    # -------------------------------------------------

    general_incidents = get_historical_incidents(
        limit=5000
    )

    # -------------------------------------------------
    # Location-aware candidate retrieval
    # -------------------------------------------------

    location_incidents = []

    if location and location.strip():
        location_incidents = (
            get_location_historical_incidents(
                location=location,
                limit=1000,
            )
        )

    print(
        "General historical incidents loaded: "
        f"{len(general_incidents)}"
    )

    print(
        "Location historical incidents loaded: "
        f"{len(location_incidents)}"
    )

    # -------------------------------------------------
    # Combine candidate pools
    # -------------------------------------------------

    combined = {}

    for incident in (
        general_incidents
        + location_incidents
    ):
        incident_id = incident.get("id")

        if incident_id is not None:
            combined[str(incident_id)] = incident

    historical_incidents = list(
        combined.values()
    )

    print(
        "Combined historical candidates: "
        f"{len(historical_incidents)}"
    )

    if not historical_incidents:
        return []

    # -------------------------------------------------
    # Existing V2 TF-IDF ranking
    # -------------------------------------------------

    return find_ai_similar_incidents(
        new_description=description,
        new_location=location,
        historical_incidents=historical_incidents,
        top_k=top_k,
    )