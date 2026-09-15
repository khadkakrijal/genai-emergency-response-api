import os
import json
import re
from collections import Counter

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REFERENCE_FILE = os.path.join(
    CURRENT_DIR,
    "train_reference.csv",
)

TEST_FILE = os.path.join(
    CURRENT_DIR,
    "test_set.csv",
)

RESULT_FILE = os.path.join(
    CURRENT_DIR,
    "threshold_evaluation_results.json",
)


# =========================================================
# CONFIGURATION
# =========================================================

SAMPLES_PER_CATEGORY = 50
TOP_K = 5
RANDOM_STATE = 42


# Thresholds to compare
THRESHOLDS = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

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

    return text


# =========================================================
# V3 REFERENCE TEXT
# =========================================================

def build_reference_text(row):
    """
    V3_with_description configuration.

    This was the best configuration from the
    previous retriever comparison experiment.
    """

    title = clean_text(
        row.get("title", "")
    )

    category = clean_text(
        row.get(
            "incident_category",
            "",
        )
    )

    description = clean_text(
        row.get("desc", "")
    )

    return (
        f"{title} "
        f"{title} "
        f"{category} "
        f"{description}"
    ).strip()


# =========================================================
# TEST QUERY
# =========================================================

def build_query(row):
    """
    Build query from held-out test case.

    The top-level category prefix is removed
    to avoid leaking the answer.

    Example:

    Traffic: VEHICLE ACCIDENT -
        ↓
    VEHICLE ACCIDENT -
    """

    title = str(
        row.get("title", "")
    ).strip()

    description = str(
        row.get("desc", "")
    ).strip()

    if ":" in title:

        title = (
            title.split(
                ":",
                1,
            )[1]
            .strip()
        )

    return clean_text(
        f"{title} {description}"
    )


# =========================================================
# CATEGORY NORMALISATION
# =========================================================

def normalise_category(value):

    if value is None:
        return "Unknown"

    value = str(
        value
    ).strip().lower()

    if value == "ems":
        return "EMS"

    if value == "traffic":
        return "Traffic"

    if value == "fire":
        return "Fire"

    return str(value)


# =========================================================
# LOAD DATA
# =========================================================

def load_reference():

    print()
    print("=" * 70)
    print("LOADING 80% REFERENCE SET")
    print("=" * 70)

    dataframe = pd.read_csv(
        REFERENCE_FILE
    )

    print(
        f"Reference incidents: "
        f"{len(dataframe):,}"
    )

    return dataframe


def load_test_sample():

    print()
    print("=" * 70)
    print("LOADING BALANCED HELD-OUT TEST SAMPLE")
    print("=" * 70)

    dataframe = pd.read_csv(
        TEST_FILE
    )

    samples = []

    for category in [
        "EMS",
        "Traffic",
        "Fire",
    ]:

        category_data = dataframe[
            dataframe[
                "incident_category"
            ] == category
        ]

        sample = category_data.sample(
            n=SAMPLES_PER_CATEGORY,
            random_state=RANDOM_STATE,
        )

        samples.append(
            sample
        )

    balanced_sample = pd.concat(
        samples,
        ignore_index=True,
    )

    balanced_sample = (
        balanced_sample
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"Evaluation cases: "
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
# PREDICT CATEGORY
# =========================================================

def predict_category(
    retrieved_categories
):

    if not retrieved_categories:
        return "Unknown"

    category_counts = Counter(
        retrieved_categories
    )

    return (
        category_counts
        .most_common(1)[0][0]
    )


# =========================================================
# BUILD TF-IDF INDEX
# =========================================================

def build_index(
    reference_dataframe
):

    print()
    print("=" * 70)
    print("BUILDING V3 TF-IDF INDEX")
    print("=" * 70)

    reference_documents = (
        reference_dataframe
        .apply(
            build_reference_text,
            axis=1,
        )
        .fillna("")
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
    )

    reference_matrix = (
        vectorizer.fit_transform(
            reference_documents
        )
    )

    print(
        f"TF-IDF matrix shape: "
        f"{reference_matrix.shape}"
    )

    return (
        vectorizer,
        reference_matrix,
    )


# =========================================================
# PRE-COMPUTE TEST RETRIEVALS
# =========================================================

def retrieve_test_cases(
    test_dataframe,
    reference_dataframe,
    vectorizer,
    reference_matrix,
):

    """
    Perform retrieval ONCE.

    Threshold experiments reuse the same retrieval
    output, so we don't repeat expensive cosine
    similarity calculations for every threshold.
    """

    print()
    print("=" * 70)
    print("PRE-COMPUTING TEST RETRIEVALS")
    print("=" * 70)

    retrieval_results = []

    total = len(
        test_dataframe
    )

    for index, row in (
        test_dataframe.iterrows()
    ):

        expected_category = (
            normalise_category(
                row[
                    "incident_category"
                ]
            )
        )

        query = build_query(
            row
        )

        query_vector = (
            vectorizer.transform(
                [query]
            )
        )

        similarity_scores = (
            cosine_similarity(
                query_vector,
                reference_matrix,
            )[0]
        )

        top_indices = (
            similarity_scores
            .argsort()[::-1][:TOP_K]
        )

        top_categories = []

        top_scores = []

        top_titles = []

        for reference_index in top_indices:

            reference_row = (
                reference_dataframe
                .iloc[
                    reference_index
                ]
            )

            category = (
                normalise_category(
                    reference_row.get(
                        "incident_category"
                    )
                )
            )

            score = float(
                similarity_scores[
                    reference_index
                ]
            )

            top_categories.append(
                category
            )

            top_scores.append(
                score
            )

            top_titles.append(
                reference_row.get(
                    "title"
                )
            )

        predicted_category = (
            predict_category(
                top_categories
            )
        )

        highest_similarity = (
            top_scores[0]
            if top_scores
            else 0.0
        )

        retrieval_results.append(
            {
                "test_number":
                    index + 1,

                "original_title":
                    row.get("title"),

                "query":
                    query,

                "expected_category":
                    expected_category,

                "predicted_category":
                    predicted_category,

                "highest_similarity":
                    highest_similarity,

                "top_categories":
                    top_categories,

                "top_scores":
                    top_scores,

                "top_titles":
                    top_titles,
            }
        )

        if (
            (index + 1) % 25 == 0
            or index == 0
        ):

            print(
                f"[{index + 1}/{total}] "
                f"Expected: "
                f"{expected_category:<8} "
                f"| Predicted: "
                f"{predicted_category:<8} "
                f"| Similarity: "
                f"{highest_similarity:.4f}"
            )

    return retrieval_results


# =========================================================
# THRESHOLD EVALUATION
# =========================================================

def evaluate_threshold(
    threshold,
    retrieval_results,
):

    expected_answered = []
    predicted_answered = []

    answered = 0
    human_review = 0

    correct_answered = 0

    false_accepts = 0
    false_rejects = 0

    all_results = []

    for result in retrieval_results:

        expected = result[
            "expected_category"
        ]

        predicted = result[
            "predicted_category"
        ]

        similarity = result[
            "highest_similarity"
        ]

        prediction_correct = (
            predicted
            ==
            expected
        )

        # =================================================
        # APPLY THRESHOLD
        # =================================================

        if similarity >= threshold:

            decision = "AI Answer"

            answered += 1

            expected_answered.append(
                expected
            )

            predicted_answered.append(
                predicted
            )

            if prediction_correct:
                correct_answered += 1

            else:
                # System confidently answered
                # but prediction was wrong.
                false_accepts += 1

        else:

            decision = "Human Review"

            human_review += 1

            # If the prediction would have been correct
            # but threshold blocks it, this is a
            # false rejection.
            if prediction_correct:
                false_rejects += 1

        all_results.append(
            {
                "test_number":
                    result[
                        "test_number"
                    ],

                "expected_category":
                    expected,

                "predicted_category":
                    predicted,

                "highest_similarity":
                    round(
                        similarity,
                        4,
                    ),

                "prediction_correct":
                    prediction_correct,

                "threshold":
                    threshold,

                "decision":
                    decision,
            }
        )

    total_cases = len(
        retrieval_results
    )

    # =====================================================
    # COVERAGE
    # =====================================================

    coverage = (
        answered
        /
        total_cases
        if total_cases
        else 0
    )

    human_review_rate = (
        human_review
        /
        total_cases
        if total_cases
        else 0
    )

    # =====================================================
    # CLASSIFICATION METRICS
    # Only cases that passed threshold
    # =====================================================

    if answered > 0:

        accuracy = accuracy_score(
            expected_answered,
            predicted_answered,
        )

        precision = precision_score(
            expected_answered,
            predicted_answered,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            expected_answered,
            predicted_answered,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            expected_answered,
            predicted_answered,
            average="weighted",
            zero_division=0,
        )

    else:

        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    # =====================================================
    # FALSE ACCEPT / REJECT RATES
    # =====================================================

    false_accept_rate = (
        false_accepts
        /
        total_cases
        if total_cases
        else 0
    )

    false_reject_rate = (
        false_rejects
        /
        total_cases
        if total_cases
        else 0
    )

    return {
        "threshold":
            threshold,

        "answered_cases":
            answered,

        "human_review_cases":
            human_review,

        "coverage":
            round(
                coverage,
                4,
            ),

        "human_review_rate":
            round(
                human_review_rate,
                4,
            ),

        "accuracy_on_answered_cases":
            round(
                accuracy,
                4,
            ),

        "precision_on_answered_cases":
            round(
                precision,
                4,
            ),

        "recall_on_answered_cases":
            round(
                recall,
                4,
            ),

        "f1_on_answered_cases":
            round(
                f1,
                4,
            ),

        "correct_answered_cases":
            correct_answered,

        "false_accepts":
            false_accepts,

        "false_accept_rate":
            round(
                false_accept_rate,
                4,
            ),

        "false_rejects":
            false_rejects,

        "false_reject_rate":
            round(
                false_reject_rate,
                4,
            ),

        "results":
            all_results,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("SIMILARITY THRESHOLD EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    reference_dataframe = (
        load_reference()
    )

    test_dataframe = (
        load_test_sample()
    )

    # -----------------------------------------------------
    # Build V3 index
    # -----------------------------------------------------

    (
        vectorizer,
        reference_matrix,
    ) = build_index(
        reference_dataframe
    )

    # -----------------------------------------------------
    # Retrieve test cases once
    # -----------------------------------------------------

    retrieval_results = (
        retrieve_test_cases(
            test_dataframe,
            reference_dataframe,
            vectorizer,
            reference_matrix,
        )
    )

    # -----------------------------------------------------
    # Evaluate thresholds
    # -----------------------------------------------------

    threshold_results = []

    print()
    print("=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            threshold,
            retrieval_results,
        )

        threshold_results.append(
            result
        )

        print()
        print(
            f"Threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"  Coverage          : "
            f"{result['coverage']:.4f}"
        )

        print(
            f"  Human Review Rate : "
            f"{result['human_review_rate']:.4f}"
        )

        print(
            f"  Accuracy           : "
            f"{result['accuracy_on_answered_cases']:.4f}"
        )

        print(
            f"  F1 Score           : "
            f"{result['f1_on_answered_cases']:.4f}"
        )

        print(
            f"  False Accepts      : "
            f"{result['false_accepts']}"
        )

        print(
            f"  False Rejects      : "
            f"{result['false_rejects']}"
        )

    # =====================================================
    # SELECT BEST THRESHOLD
    # =====================================================

    """
    We want a balance between:
    - high F1
    - reasonable coverage
    - low false accepts

    This simple research score combines
    F1 and coverage.
    """

    for result in threshold_results:

        result[
            "selection_score"
        ] = round(
            result[
                "f1_on_answered_cases"
            ]
            *
            result[
                "coverage"
            ],
            4,
        )

    best_threshold_result = max(
        threshold_results,
        key=lambda item:
            item[
                "selection_score"
            ],
    )

    print()
    print("=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{best_threshold_result['threshold']}"
    )

    print(
        f"Coverage: "
        f"{best_threshold_result['coverage']}"
    )

    print(
        f"F1 Score: "
        f"{best_threshold_result['f1_on_answered_cases']}"
    )

    print(
        f"False Accepts: "
        f"{best_threshold_result['false_accepts']}"
    )

    print(
        f"False Rejects: "
        f"{best_threshold_result['false_rejects']}"
    )

    print(
        f"Selection Score: "
        f"{best_threshold_result['selection_score']}"
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output = {
        "experiment":
            "Similarity threshold evaluation",

        "retriever_configuration":
            "V3_with_description",

        "reference_set":
            "80%",

        "held_out_test_set":
            "20%",

        "evaluation_cases":
            len(
                test_dataframe
            ),

        "top_k":
            TOP_K,

        "thresholds_tested":
            THRESHOLDS,

        "best_threshold":
            best_threshold_result[
                "threshold"
            ],

        "best_threshold_summary":
            {
                key: value
                for key, value
                in best_threshold_result.items()
                if key != "results"
            },

        "threshold_results":
            [
                {
                    key: value
                    for key, value
                    in result.items()
                    if key != "results"
                }

                for result
                in threshold_results
            ],
    }

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("THRESHOLD EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )


if __name__ == "__main__":

    main()