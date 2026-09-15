import os
import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATH
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REFERENCE_FILE = os.path.join(
    CURRENT_DIR,
    "train_reference.csv",
)


# =========================================================
# DOMAIN VOCABULARY
# =========================================================

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
    "house fire": "building fire",
    "kitchen fire": "building fire",

    # Medical
    "dizzy": "dizziness ems medical",
    "dizziness": "dizziness ems medical",
    "unconscious": "ems medical emergency",
    "chest pain": "cardiac emergency ems",
    "heart attack": "cardiac emergency ems",
    "difficulty breathing": "respiratory emergency ems",

    # Violence
    "knife": "weapon assault victim",
    "threatening": "assault victim threat",
    "threat": "assault victim threat",
    "attacked": "assault victim",
    "assaulted": "assault victim",
}


# =========================================================
# TEXT NORMALISATION
# =========================================================

def normalise_text(text: str) -> str:
    """
    Clean and expand emergency-related terminology.
    """

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s/-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    expanded = [text]

    for phrase, expansion in DOMAIN_EXPANSIONS.items():

        if phrase in text:
            expanded.append(expansion)

    return " ".join(expanded)


# =========================================================
# REFERENCE RAG RETRIEVER
# =========================================================

class ReferenceRetriever:
    """
    Evaluation-only RAG retriever.

    Uses ONLY the 80% train_reference.csv dataset.

    The 20% test_set.csv is never used as retrieval
    knowledge, preventing train/test leakage.
    """

    def __init__(self):

        print()
        print("=" * 70)
        print("INITIALISING 80% REFERENCE RETRIEVER")
        print("=" * 70)

        self.dataframe = None
        self.vectorizer = None
        self.reference_matrix = None

        self._load_reference_data()
        self._build_tfidf_index()

    # =====================================================
    # LOAD 80% REFERENCE DATA
    # =====================================================

    def _load_reference_data(self):

        if not os.path.exists(
            REFERENCE_FILE
        ):
            raise FileNotFoundError(
                f"Reference dataset not found:\n"
                f"{REFERENCE_FILE}\n\n"
                "Run split_dataset.py first."
            )

        print()
        print("Loading reference dataset:")
        print(REFERENCE_FILE)

        self.dataframe = pd.read_csv(
            REFERENCE_FILE
        )

        print()
        print(
            f"Reference incidents loaded: "
            f"{len(self.dataframe):,}"
        )

        required_columns = [
            "title",
            "desc",
            "incident_category",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column
            not in self.dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

    # =====================================================
    # CREATE SEARCHABLE TEXT
    # =====================================================

    def _create_searchable_text(
        self,
        row,
    ):
        """
        Kaggle 911 contains most useful emergency meaning
        in the title.

        Example:
        EMS: DIZZINESS
        Traffic: VEHICLE ACCIDENT -
        Fire: BUILDING FIRE
        """

        title = normalise_text(
            row.get("title", "")
        )

        category = normalise_text(
            row.get(
                "incident_category",
                "",
            )
        )

        # Repeat title to give emergency label more weight.
        return (
            f"{title} "
            f"{title} "
            f"{title} "
            f"{category}"
        ).strip()

    # =====================================================
    # BUILD TF-IDF INDEX ONCE
    # =====================================================

    def _build_tfidf_index(self):

        print()
        print(
            "Building TF-IDF reference index..."
        )

        searchable_documents = (
            self.dataframe.apply(
                self._create_searchable_text,
                axis=1,
            )
            .fillna("")
            .tolist()
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
        )

        self.reference_matrix = (
            self.vectorizer.fit_transform(
                searchable_documents
            )
        )

        print()
        print(
            "TF-IDF reference index created."
        )

        print(
            f"Matrix shape: "
            f"{self.reference_matrix.shape}"
        )

    # =====================================================
    # RETRIEVE TOP-K
    # =====================================================

    def retrieve(
        self,
        description: str,
        top_k: int = 5,
    ):

        if not description:
            return []

        query = normalise_text(
            description
        )

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        similarity_scores = cosine_similarity(
            query_vector,
            self.reference_matrix,
        )[0]

        ranked_indices = (
            similarity_scores
            .argsort()[::-1][:top_k]
        )

        results = []

        for index in ranked_indices:

            row = self.dataframe.iloc[
                index
            ]

            score = float(
                similarity_scores[index]
            )

            results.append(
                {
                    "reference_index":
                        int(index),

                    "title":
                        row.get("title"),

                    "description":
                        row.get("desc"),

                    "location":
                        row.get("addr"),

                    "township":
                        row.get("twp"),

                    "incident_type":
                        row.get(
                            "incident_category"
                        ),

                    "latitude":
                        row.get("lat"),

                    "longitude":
                        row.get("lng"),

                    "incident_time":
                        row.get("timeStamp"),

                    "similarity_score":
                        round(
                            score,
                            4,
                        ),
                }
            )

        return results


# =========================================================
# OPTIONAL QUICK TEST
# =========================================================

if __name__ == "__main__":

    retriever = ReferenceRetriever()

    test_query = (
        "Two vehicles collided and "
        "one person was injured."
    )

    print()
    print("=" * 70)
    print("QUICK RETRIEVAL TEST")
    print("=" * 70)

    print()
    print(
        f"Query: {test_query}"
    )

    results = retriever.retrieve(
        test_query,
        top_k=5,
    )

    print()

    for position, incident in enumerate(
        results,
        start=1,
    ):

        print(
            f"{position}. "
            f"{incident['title']} "
            f"| Category: "
            f"{incident['incident_type']} "
            f"| Similarity: "
            f"{incident['similarity_score']}"
        )