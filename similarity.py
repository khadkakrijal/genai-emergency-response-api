from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")


def find_ai_similar_incidents(new_description: str, historical_incidents: list, top_k: int = 3):
    if not historical_incidents:
        return []

    historical_texts = [
        f"{item.get('title', '')} {item.get('description', '')}"
        for item in historical_incidents
    ]

    new_embedding = model.encode(new_description, convert_to_tensor=True)
    historical_embeddings = model.encode(historical_texts, convert_to_tensor=True)

    scores = util.cos_sim(new_embedding, historical_embeddings)[0]

    top_results = scores.topk(k=min(top_k, len(historical_incidents)))

    similar = []

    for score, idx in zip(top_results.values, top_results.indices):
        item = historical_incidents[int(idx)]

        similar.append({
            "title": item.get("title"),
            "description": item.get("description"),
            "location": item.get("location"),
            "incident_type": item.get("incident_type"),
            "priority_level": item.get("priority_level"),
            "similarity_score": round(float(score), 3),
        })

    return similar