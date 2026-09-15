import json
import os
from collections import Counter

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from ai.evaluation.reference_retriever import ReferenceRetriever


# =========================================================
# CONFIGURATION
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_FILE = os.path.join(
    CURRENT_DIR,
    "test_set.csv",
)

RESULT_JSON_FILE = os.path.join(
    CURRENT_DIR,
    "holdout_evaluation_results.json",
)

RESULT_CSV_FILE = os.path.join(
    CURRENT_DIR,
    "holdout_predictions.csv",
)


# Number of unseen examples per category
SAMPLES_PER_CATEGORY = 50

TOP_K = 5

RANDOM_STATE = 42


# =========================================================
# NORMALISE CATEGORY
# =========================================================

def normalise_category(value):
    """
    Convert category names into a consistent format.
    """

    if value is None:
        return "Unknown"

    value = str(value).strip().lower()

    if value == "ems":
        return "EMS"

    if value == "traffic":
        return "Traffic"

    if value == "fire":
        return "Fire"

    return str(value).strip()


# =========================================================
# BUILD TEST QUERY
# =========================================================

def build_test_query(row):
    """
    Build a query for the RAG retriever from the held-out
    Kaggle test incident.

    IMPORTANT:
    We remove the top-level category prefix from title.

    Example:

    Traffic: VEHICLE ACCIDENT -
                ↓
    VEHICLE ACCIDENT -

    This prevents directly giving the retriever the target
    class name 'Traffic'.
    """

    title = str(
        row.get("title", "")
    ).strip()

    description = str(
        row.get("desc", "")
    ).strip()

    # Remove category prefix such as:
    # EMS:
    # Traffic:
    # Fire:
    if ":" in title:
        title_without_category = (
            title.split(
                ":",
                1,
            )[1]
            .strip()
        )
    else:
        title_without_category = title

    # The Kaggle description mostly contains location/time,
    # but we include it as secondary contextual information.
    query = (
        f"{title_without_category} "
        f"{description}"
    ).strip()

    return query


# =========================================================
# LOAD HELD-OUT TEST SET
# =========================================================

def load_test_set():

    if not os.path.exists(TEST_FILE):

        raise FileNotFoundError(
            f"Test dataset not found:\n"
            f"{TEST_FILE}\n\n"
            "Run split_dataset.py first."
        )

    print()
    print("=" * 70)
    print("LOADING 20% HELD-OUT TEST SET")
    print("=" * 70)

    dataframe = pd.read_csv(
        TEST_FILE
    )

    print()
    print(
        f"Total held-out incidents: "
        f"{len(dataframe):,}"
    )

    print()
    print(
        dataframe[
            "incident_category"
        ].value_counts()
    )

    return dataframe


# =========================================================
# CREATE BALANCED TEST SAMPLE
# =========================================================

def create_balanced_sample(dataframe):
    """
    Select an equal number of EMS, Traffic and Fire cases.

    This avoids the larger EMS class dominating the
    initial pilot evaluation.
    """

    samples = []

    categories = [
        "EMS",
        "Traffic",
        "Fire",
    ]

    for category in categories:

        category_data = dataframe[
            dataframe[
                "incident_category"
            ] == category
        ]

        if (
            len(category_data)
            <
            SAMPLES_PER_CATEGORY
        ):

            raise ValueError(
                f"Not enough {category} rows "
                f"for sampling."
            )

        sample = category_data.sample(
            n=SAMPLES_PER_CATEGORY,
            random_state=RANDOM_STATE,
        )

        samples.append(sample)

    balanced_sample = pd.concat(
        samples,
        ignore_index=True,
    )

    # Shuffle after combining categories
    balanced_sample = (
        balanced_sample
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("BALANCED TEST SAMPLE")
    print("=" * 70)

    print(
        f"Total evaluation cases: "
        f"{len(balanced_sample)}"
    )

    print()

    print(
        balanced_sample[
            "incident_category"
        ].value_counts()
    )

    return balanced_sample


# =========================================================
# PREDICT CATEGORY FROM TOP-K
# =========================================================

def predict_category(
    retrieved_incidents
):
    """
    Predict the category using majority voting over
    the Top-K retrieved historical incidents.

    Example:

    Traffic
    Traffic
    Traffic
    EMS
    Traffic

    Prediction = Traffic
    """

    if not retrieved_incidents:
        return "Unknown"

    categories = []

    for incident in retrieved_incidents:

        category = normalise_category(
            incident.get(
                "incident_type"
            )
        )

        categories.append(category)

    category_counts = Counter(
        categories
    )

    predicted_category = (
        category_counts
        .most_common(1)[0][0]
    )

    return predicted_category


# =========================================================
# CALCULATE PRECISION@K
# =========================================================

def calculate_precision_at_k(
    retrieved_incidents,
    expected_category,
    k=5,
):
    """
    Precision@K measures how many of the Top-K retrieved
    historical incidents belong to the correct category.
    """

    top_results = (
        retrieved_incidents[:k]
    )

    if not top_results:
        return 0.0

    relevant_count = 0

    for incident in top_results:

        retrieved_category = (
            normalise_category(
                incident.get(
                    "incident_type"
                )
            )
        )

        if (
            retrieved_category
            ==
            expected_category
        ):
            relevant_count += 1

    precision_at_k = (
        relevant_count
        /
        len(top_results)
    )

    return round(
        precision_at_k,
        4,
    )


# =========================================================
# RUN HOLDOUT EVALUATION
# =========================================================

def run_evaluation():

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    full_test_set = load_test_set()

    test_sample = (
        create_balanced_sample(
            full_test_set
        )
    )

    # -----------------------------------------------------
    # Initialise 80% RAG reference retriever ONCE
    # -----------------------------------------------------

    retriever = ReferenceRetriever()

    expected_labels = []
    predicted_labels = []

    precision_at_5_scores = []

    results = []

    total_cases = len(
        test_sample
    )

    print()
    print("=" * 70)
    print("STARTING 80/20 HOLDOUT EVALUATION")
    print("=" * 70)

    # =====================================================
    # PROCESS EACH TEST CASE
    # =====================================================

    for row_number, row in (
        test_sample.iterrows()
    ):

        test_number = (
            row_number + 1
        )

        expected_category = (
            normalise_category(
                row[
                    "incident_category"
                ]
            )
        )

        query = build_test_query(
            row
        )

        # -------------------------------------------------
        # Retrieve ONLY from 80% reference set
        # -------------------------------------------------

        retrieved_incidents = (
            retriever.retrieve(
                description=query,
                top_k=TOP_K,
            )
        )

        # -------------------------------------------------
        # Predict category
        # -------------------------------------------------

        predicted_category = (
            predict_category(
                retrieved_incidents
            )
        )

        # -------------------------------------------------
        # Precision@5
        # -------------------------------------------------

        precision_at_5 = (
            calculate_precision_at_k(
                retrieved_incidents,
                expected_category,
                k=TOP_K,
            )
        )

        # -------------------------------------------------
        # Highest similarity
        # -------------------------------------------------

        if retrieved_incidents:

            highest_similarity = (
                retrieved_incidents[0]
                .get(
                    "similarity_score",
                    0,
                )
            )

        else:

            highest_similarity = 0

        # -------------------------------------------------
        # Save labels
        # -------------------------------------------------

        expected_labels.append(
            expected_category
        )

        predicted_labels.append(
            predicted_category
        )

        precision_at_5_scores.append(
            precision_at_5
        )

        correct = (
            expected_category
            ==
            predicted_category
        )

        # -------------------------------------------------
        # Progress output
        # -------------------------------------------------

        print(
            f"[{test_number}/{total_cases}] "
            f"Expected: {expected_category:<8} "
            f"| Predicted: {predicted_category:<8} "
            f"| Correct: {correct} "
            f"| P@5: {precision_at_5:.2f} "
            f"| Similarity: {highest_similarity}"
        )

        # -------------------------------------------------
        # Store detailed result
        # -------------------------------------------------

        results.append(
            {
                "test_number":
                    test_number,

                "original_title":
                    row.get(
                        "title"
                    ),

                "query":
                    query,

                "expected_category":
                    expected_category,

                "predicted_category":
                    predicted_category,

                "correct":
                    correct,

                "precision_at_5":
                    precision_at_5,

                "highest_similarity":
                    highest_similarity,

                "top_5_titles": [
                    incident.get(
                        "title"
                    )
                    for incident
                    in retrieved_incidents
                ],

                "top_5_categories": [
                    incident.get(
                        "incident_type"
                    )
                    for incident
                    in retrieved_incidents
                ],
            }
        )

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================

    accuracy = accuracy_score(
        expected_labels,
        predicted_labels,
    )

    precision = precision_score(
        expected_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        expected_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        expected_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    # =====================================================
    # RETRIEVAL METRIC
    # =====================================================

    mean_precision_at_5 = (
        sum(
            precision_at_5_scores
        )
        /
        len(
            precision_at_5_scores
        )
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    categories = [
        "EMS",
        "Traffic",
        "Fire",
    ]

    report = classification_report(
        expected_labels,
        predicted_labels,
        labels=categories,
        output_dict=True,
        zero_division=0,
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    matrix = confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=categories,
    )

    # =====================================================
    # PRINT FINAL RESULTS
    # =====================================================

    print()
    print("=" * 70)
    print("HOLDOUT CLASSIFICATION PERFORMANCE")
    print("=" * 70)

    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )

    print()
    print("=" * 70)
    print("RAG RETRIEVAL PERFORMANCE")
    print("=" * 70)

    print(
        f"Mean Precision@{TOP_K}: "
        f"{mean_precision_at_5:.4f}"
    )

    print()
    print("=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)

    for category in categories:

        category_metrics = (
            report.get(
                category,
                {},
            )
        )

        print()
        print(
            f"{category}:"
        )

        print(
            f"  Precision: "
            f"{category_metrics.get('precision', 0):.4f}"
        )

        print(
            f"  Recall   : "
            f"{category_metrics.get('recall', 0):.4f}"
        )

        print(
            f"  F1 Score : "
            f"{category_metrics.get('f1-score', 0):.4f}"
        )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print()
    print(
        "Labels:"
    )

    print(
        categories
    )

    print()

    print(
        matrix
    )

    # =====================================================
    # SAVE JSON RESULT
    # =====================================================

    summary = {
        "evaluation": {
            "type":
                "80_20_holdout",

            "reference_set":
                "80%",

            "test_set":
                "20%",

            "sample_size":
                total_cases,

            "samples_per_category":
                SAMPLES_PER_CATEGORY,

            "top_k":
                TOP_K,

            "random_state":
                RANDOM_STATE,
        },

        "classification_metrics": {
            "accuracy":
                round(
                    accuracy,
                    4,
                ),

            "precision":
                round(
                    precision,
                    4,
                ),

            "recall":
                round(
                    recall,
                    4,
                ),

            "f1_score":
                round(
                    f1,
                    4,
                ),
        },

        "retrieval_metrics": {
            f"mean_precision_at_{TOP_K}":
                round(
                    mean_precision_at_5,
                    4,
                )
        },

        "per_class_metrics":
            report,

        "confusion_matrix": {
            "labels":
                categories,

            "matrix":
                matrix.tolist(),
        },

        "results":
            results,
    }

    with open(
        RESULT_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # =====================================================
    # SAVE CSV
    # =====================================================

    result_dataframe = (
        pd.DataFrame(
            results
        )
    )

    result_dataframe.to_csv(
        RESULT_CSV_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "JSON results saved to:"
    )

    print(
        RESULT_JSON_FILE
    )

    print()
    print(
        "CSV predictions saved to:"
    )

    print(
        RESULT_CSV_FILE
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_evaluation()