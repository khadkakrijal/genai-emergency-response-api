import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Emergency-domain vocabulary normalisation
# ---------------------------------------------------------

DOMAIN_EXPANSIONS = {
    # Traffic
    "collision": "vehicle accident traffic",
    "collided": "vehicle accident traffic",
    "crash": "vehicle accident traffic",
    "car accident": "vehicle accident traffic",
    "vehicle collision": "vehicle accident traffic",

    # Fire
    "smoke": "fire burning",
    "flame": "fire burning",
    "flames": "fire burning",
    "burning": "fire",

    # Medical
    "dizzy": "dizziness ems medical",
    "dizziness": "dizziness ems medical",
    "unconscious": "ems medical emergency",
    "chest pain": "cardiac emergency ems",
    "heart attack": "cardiac emergency ems",
    "difficulty breathing": "respiratory emergency ems",

    # Violence / threat
    "knife": "weapon assault threat",
    "threatening": "threat assault",
    "threat": "assault threat",
    "attacked": "assault victim",
    "assaulted": "assault victim",
}


def normalise_text(text: str) -> str:
    """
    Normalise natural-language incident descriptions and
    expand common emergency vocabulary into terms that
    occur in the historical 911 dataset.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Basic cleanup
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    expanded_terms = [text]

    for phrase, expansion in DOMAIN_EXPANSIONS.items():
        if phrase in text:
            expanded_terms.append(expansion)

    return " ".join(expanded_terms)


def build_historical_text(incident: dict):
    """
    Build searchable historical text.

    The Kaggle dataset stores the useful emergency meaning
    mainly in the title and incident_type fields.
    """

    title = normalise_text(
        incident.get("title") or ""
    )

    incident_type = normalise_text(
        incident.get("incident_type") or ""
    )

    return (
        f"{title} "
        f"{title} "
        f"{title} "
        f"{title} "
        f"{incident_type}"
    ).strip()


def find_ai_similar_incidents(
    new_description: str,
    historical_incidents: list,
    top_k: int = 5,
):
    """
    Retrieve similar incidents using an ensemble of:

    1. Word-level TF-IDF
    2. Character-level TF-IDF
    3. Cosine Similarity

    Character n-grams help with related forms such as
    'dizzy' and 'dizziness'.
    """

    if not new_description or not new_description.strip():
        return []

    if not historical_incidents:
        return []

    query = normalise_text(new_description)

    documents = []
    valid_incidents = []

    for incident in historical_incidents:
        searchable_text = build_historical_text(
            incident
        )

        if not searchable_text:
            continue

        documents.append(searchable_text)
        valid_incidents.append(incident)

    if not documents:
        return []

    try:
        all_documents = [query] + documents

        # -------------------------------------------------
        # Word-level TF-IDF
        # -------------------------------------------------

        word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
        )

        word_matrix = word_vectorizer.fit_transform(
            all_documents
        )

        word_scores = cosine_similarity(
            word_matrix[0:1],
            word_matrix[1:],
        )[0]

        # -------------------------------------------------
        # Character-level TF-IDF
        # -------------------------------------------------

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )

        char_matrix = char_vectorizer.fit_transform(
            all_documents
        )

        char_scores = cosine_similarity(
            char_matrix[0:1],
            char_matrix[1:],
        )[0]

        # -------------------------------------------------
        # Combined score
        # -------------------------------------------------

        combined_scores = (
            0.70 * word_scores
            +
            0.30 * char_scores
        )

        ranked_indices = combined_scores.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            incident = valid_incidents[index]

            score = float(
                combined_scores[index]
            )

            results.append(
                {
                    "id": incident.get("id"),
                    "title": incident.get("title"),
                    "description": incident.get(
                        "description"
                    ),
                    "location": incident.get(
                        "location"
                    ),
                    "incident_type": incident.get(
                        "incident_type"
                    ),
                    "priority_level": incident.get(
                        "priority_level"
                    ),

                    "similarity_score": round(
                        score,
                        4,
                    ),

                    # Useful for research/debugging
                    "word_similarity": round(
                        float(word_scores[index]),
                        4,
                    ),

                    "character_similarity": round(
                        float(char_scores[index]),
                        4,
                    ),
                }
            )

        return results

    except Exception as error:
        print(
            f"Similarity calculation error: {error}"
        )
        return []