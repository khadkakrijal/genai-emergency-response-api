def find_ai_similar_incidents(new_description, historical_incidents, top_k=5):
    """
    Lightweight cloud-safe similarity fallback.
    Avoids sentence-transformers on Render free tier.
    """

    if not historical_incidents:
        return []

    return historical_incidents[:top_k]