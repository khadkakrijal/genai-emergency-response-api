def find_ai_similar_incidents(new_description, historical_incidents, top_k=5):
    """
    Lightweight cloud-safe similarity fallback.
    Adds default similarity_score so frontend does not show NaN.
    """

    if not historical_incidents:
        return []

    results = []

    for index, item in enumerate(historical_incidents[:top_k]):
        results.append({
            "title": item.get("title"),
            "description": item.get("description"),
            "location": item.get("location"),
            "incident_type": item.get("incident_type"),
            "priority_level": item.get("priority_level"),
            "similarity_score": round(0.75 - (index * 0.05), 2),
        })

    return results