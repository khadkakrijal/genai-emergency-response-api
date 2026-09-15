import os
import re
import json
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
    "retriever_comparison_results.json",
)


# =========================================================
# CONFIGURATION
# =========================================================

SAMPLES_PER_CATEGORY = 50
TOP_K = 5
RANDOM_STATE = 42


# =========================================================
# TEXT NORMALISATION
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
# LOAD DATA
# =========================================================

def load_reference():

    print()
    print("=" * 70)
    print("LOADING 80% REFERENCE SET")
    print("=" * 70)

    df = pd.read_csv(
        REFERENCE_FILE
    )

    print(
        f"Reference incidents: "
        f"{len(df):,}"
    )

    return df


def load_test_sample():

    print()
    print("=" * 70)
    print("LOADING 20% TEST SET")
    print("=" * 70)

    df = pd.read_csv(
        TEST_FILE
    )

    samples = []

    for category in [
        "EMS",
        "Traffic",
        "Fire",
    ]:

        category_df = df[
            df["incident_category"]
            ==
            category
        ]

        sample = category_df.sample(
            n=SAMPLES_PER_CATEGORY,
            random_state=RANDOM_STATE,
        )

        samples.append(sample)

    balanced = pd.concat(
        samples,
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(
        f"Evaluation sample: "
        f"{len(balanced)}"
    )

    print()

    print(
        balanced[
            "incident_category"
        ].value_counts()
    )

    return balanced


# =========================================================
# BUILD QUERY FROM TEST CASE
# =========================================================

def build_query(row):

    title = str(
        row.get("title", "")
    )

    desc = str(
        row.get("desc", "")
    )

    # Remove target class prefix
    # Example:
    # Traffic: VEHICLE ACCIDENT -
    # becomes:
    # VEHICLE ACCIDENT -
    if ":" in title:

        title = (
            title.split(
                ":",
                1,
            )[1]
            .strip()
        )

    return clean_text(
        f"{title} {desc}"
    )


# =========================================================
# REFERENCE TEXT CONFIGURATIONS
# =========================================================

def build_reference_text_v1(row):
    """
    V1 - current baseline style.
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

    return (
        f"{title} "
        f"{title} "
        f"{title} "
        f"{category}"
    ).strip()


def build_reference_text_v2(row):
    """
    V2 - stronger title weighting.
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

    return (
        f"{title} "
        f"{title} "
        f"{title} "
        f"{title} "
        f"{title} "
        f"{category}"
    ).strip()


def build_reference_text_v3(row):
    """
    V3 - title + category + original description.
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

    desc = clean_text(
        row.get("desc", "")
    )

    return (
        f"{title} "
        f"{title} "
        f"{category} "
        f"{desc}"
    ).strip()


def build_reference_text_v4(row):
    """
    V4 - strongly weighted title + description.
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

    desc = clean_text(
        row.get("desc", "")
    )

    return (
        f"{title} "
        f"{title} "
        f"{title} "
        f"{title} "
        f"{category} "
        f"{desc}"
    ).strip()


# =========================================================
# TF-IDF CONFIGURATIONS
# =========================================================

CONFIGURATIONS = {
    "V1_baseline": {
        "builder":
            build_reference_text_v1,

        "vectorizer":
            {
                "ngram_range":
                    (1, 2),

                "max_features":
                    30000,
            },
    },

    "V2_title_weighted": {
        "builder":
            build_reference_text_v2,

        "vectorizer":
            {
                "ngram_range":
                    (1, 2),

                "max_features":
                    30000,
            },
    },

    "V3_with_description": {
        "builder":
            build_reference_text_v3,

        "vectorizer":
            {
                "ngram_range":
                    (1, 2),

                "max_features":
                    40000,
            },
    },

    "V4_title_weighted_description": {
        "builder":
            build_reference_text_v4,

        "vectorizer":
            {
                "ngram_range":
                    (1, 3),

                "max_features":
                    50000,
            },
    },
}


# =========================================================
# CATEGORY NORMALISATION
# =========================================================

def normalise_category(value):

    value = str(
        value
    ).strip().lower()

    if value == "ems":
        return "EMS"

    if value == "traffic":
        return "Traffic"

    if value == "fire":
        return "Fire"

    return value


# =========================================================
# PREDICT CATEGORY
# =========================================================

def predict_category(
    retrieved_categories
):

    if not retrieved_categories:
        return "Unknown"

    counts = Counter(
        retrieved_categories
    )

    return (
        counts
        .most_common(1)[0][0]
    )


# =========================================================
# RUN ONE CONFIGURATION
# =========================================================

def evaluate_configuration(
    name,
    configuration,
    reference_df,
    test_df,
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING {name}"
    )
    print("=" * 70)

    builder = configuration[
        "builder"
    ]

    vectorizer_options = configuration[
        "vectorizer"
    ]

    print()
    print(
        "Building reference text..."
    )

    reference_documents = (
        reference_df.apply(
            builder,
            axis=1,
        )
        .fillna("")
        .tolist()
    )

    print(
        "Building TF-IDF matrix..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=
            vectorizer_options[
                "ngram_range"
            ],

        max_features=
            vectorizer_options[
                "max_features"
            ],

        sublinear_tf=True,
    )

    reference_matrix = (
        vectorizer.fit_transform(
            reference_documents
        )
    )

    print(
        f"Matrix shape: "
        f"{reference_matrix.shape}"
    )

    expected_labels = []
    predicted_labels = []

    precision_at_k_scores = []

    incorrect_count = 0

    total = len(
        test_df
    )

    # =====================================================
    # PROCESS TEST CASES
    # =====================================================

    for index, row in (
        test_df.iterrows()
    ):

        expected = normalise_category(
            row[
                "incident_category"
            ]
        )

        query = build_query(
            row
        )

        query_vector = (
            vectorizer.transform(
                [query]
            )
        )

        similarities = (
            cosine_similarity(
                query_vector,
                reference_matrix,
            )[0]
        )

        top_indices = (
            similarities
            .argsort()[::-1][:TOP_K]
        )

        retrieved_categories = []

        for ref_index in top_indices:

            category = (
                reference_df
                .iloc[ref_index]
                .get(
                    "incident_category"
                )
            )

            category = (
                normalise_category(
                    category
                )
            )

            retrieved_categories.append(
                category
            )

        predicted = predict_category(
            retrieved_categories
        )

        expected_labels.append(
            expected
        )

        predicted_labels.append(
            predicted
        )

        relevant = sum(
            1
            for category
            in retrieved_categories
            if category == expected
        )

        precision_at_k = (
            relevant
            /
            TOP_K
        )

        precision_at_k_scores.append(
            precision_at_k
        )

        if predicted != expected:
            incorrect_count += 1

        # Keep terminal readable
        if (
            (index + 1) % 25 == 0
            or index == 0
        ):

            print(
                f"[{index + 1}/{total}] "
                f"Expected: {expected:<8} "
                f"| Predicted: {predicted:<8} "
                f"| P@{TOP_K}: "
                f"{precision_at_k:.2f}"
            )

    # =====================================================
    # METRICS
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

    mean_precision_at_k = (
        sum(
            precision_at_k_scores
        )
        /
        len(
            precision_at_k_scores
        )
    )

    print()
    print("-" * 70)

    print(
        f"{name} RESULTS"
    )

    print("-" * 70)

    print(
        f"Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision      : "
        f"{precision:.4f}"
    )

    print(
        f"Recall         : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score       : "
        f"{f1:.4f}"
    )

    print(
        f"Mean P@{TOP_K:<9}: "
        f"{mean_precision_at_k:.4f}"
    )

    print(
        f"Incorrect Cases: "
        f"{incorrect_count}"
    )

    return {
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

        f"mean_precision_at_{TOP_K}":
            round(
                mean_precision_at_k,
                4,
            ),

        "incorrect_cases":
            incorrect_count,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "TF-IDF RETRIEVER COMPARISON"
    )
    print("=" * 70)

    reference_df = (
        load_reference()
    )

    test_df = (
        load_test_sample()
    )

    comparison_results = {}

    for (
        configuration_name,
        configuration,
    ) in CONFIGURATIONS.items():

        result = (
            evaluate_configuration(
                configuration_name,
                configuration,
                reference_df,
                test_df,
            )
        )

        comparison_results[
            configuration_name
        ] = result

    # =====================================================
    # FIND BEST CONFIGURATION
    # =====================================================

    best_configuration = max(
        comparison_results,
        key=lambda key:
            comparison_results[
                key
            ][
                "f1_score"
            ],
    )

    print()
    print("=" * 70)
    print(
        "FINAL COMPARISON"
    )
    print("=" * 70)

    for name, metrics in (
        comparison_results.items()
    ):

        print()
        print(name)

        print(
            f"  Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall   : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1       : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"  P@{TOP_K}      : "
            f"{metrics[f'mean_precision_at_{TOP_K}']:.4f}"
        )

        print(
            f"  Errors   : "
            f"{metrics['incorrect_cases']}"
        )

    print()
    print("=" * 70)

    print(
        f"BEST CONFIGURATION: "
        f"{best_configuration}"
    )

    print("=" * 70)

    # =====================================================
    # SAVE
    # =====================================================

    output = {
        "experiment":
            "TF-IDF retrieval configuration comparison",

        "reference_records":
            len(
                reference_df
            ),

        "test_cases":
            len(
                test_df
            ),

        "top_k":
            TOP_K,

        "random_state":
            RANDOM_STATE,

        "best_configuration":
            best_configuration,

        "results":
            comparison_results,
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
    print(
        "Results saved to:"
    )

    print(
        RESULT_FILE
    )


if __name__ == "__main__":

    main()