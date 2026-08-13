from database import get_historical_incidents
from similarity import find_ai_similar_incidents


def retrieve_similar_incidents(
    description: str,
    top_k: int = 5,
):
    if not description or not description.strip():
        return []

    historical_incidents = get_historical_incidents(
        limit=5000
    )

    print(
        f"Historical incidents loaded: "
        f"{len(historical_incidents)}"
    )

    if not historical_incidents:
        return []

    return find_ai_similar_incidents(
        new_description=description,
        historical_incidents=historical_incidents,
        top_k=top_k,
    )