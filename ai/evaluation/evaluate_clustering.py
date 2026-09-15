import os
import re
import json

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
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

RESULT_JSON_FILE = os.path.join(
    CURRENT_DIR,
    "clustering_evaluation_results.json",
)

RESULT_CSV_FILE = os.path.join(
    CURRENT_DIR,
    "clustering_comparison.csv",
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# 1000 EMS + 1000 Traffic + 1000 Fire
SAMPLES_PER_CATEGORY = 1000

# Dimensionality after TF-IDF
SVD_COMPONENTS = 100

# Cluster counts to compare
CLUSTER_COUNTS = [
    2,
    3,
    4,
    5,
    6,
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
# V3 TEXT REPRESENTATION
# =========================================================

def build_reference_text(row):
    """
    Same V3 representation selected as the best
    retriever configuration in the previous experiment.

    V3:
        title + title + category + description
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
# LOAD BALANCED SAMPLE
# =========================================================

def load_balanced_sample():

    print()
    print("=" * 70)
    print("LOADING 80% REFERENCE SET")
    print("=" * 70)

    dataframe = pd.read_csv(
        REFERENCE_FILE
    )

    print(
        f"Reference incidents available: "
        f"{len(dataframe):,}"
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

    print()
    print("=" * 70)
    print("BALANCED CLUSTERING SAMPLE")
    print("=" * 70)

    print(
        f"Total clustering samples: "
        f"{len(balanced_sample):,}"
    )

    print()

    print(
        balanced_sample[
            "incident_category"
        ].value_counts()
    )

    return balanced_sample


# =========================================================
# BUILD TF-IDF REPRESENTATION
# =========================================================

def build_tfidf_matrix(
    dataframe
):

    print()
    print("=" * 70)
    print("BUILDING V3 TF-IDF REPRESENTATION")
    print("=" * 70)

    documents = (
        dataframe
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

    tfidf_matrix = (
        vectorizer.fit_transform(
            documents
        )
    )

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    return tfidf_matrix


# =========================================================
# DIMENSIONALITY REDUCTION
# =========================================================

def reduce_dimensions(
    tfidf_matrix
):

    print()
    print("=" * 70)
    print("APPLYING TRUNCATED SVD")
    print("=" * 70)

    print(
        f"Reducing TF-IDF vectors to "
        f"{SVD_COMPONENTS} dimensions..."
    )

    svd = TruncatedSVD(
        n_components=SVD_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    reduced_matrix = (
        svd.fit_transform(
            tfidf_matrix
        )
    )

    # Normalise reduced vectors
    normalizer = Normalizer(
        copy=False
    )

    reduced_matrix = (
        normalizer.fit_transform(
            reduced_matrix
        )
    )

    explained_variance = (
        svd.explained_variance_ratio_
        .sum()
    )

    print()
    print(
        f"Reduced matrix shape: "
        f"{reduced_matrix.shape}"
    )

    print(
        f"Explained variance: "
        f"{explained_variance:.4f}"
    )

    return (
        reduced_matrix,
        explained_variance,
    )


# =========================================================
# CLUSTER COMPOSITION
# =========================================================

def calculate_cluster_composition(
    dataframe,
    labels,
    number_of_clusters,
):

    temp = dataframe[
        [
            "incident_category"
        ]
    ].copy()

    temp[
        "cluster"
    ] = labels

    composition = {}

    for cluster_id in range(
        number_of_clusters
    ):

        cluster_rows = temp[
            temp[
                "cluster"
            ] == cluster_id
        ]

        category_counts = (
            cluster_rows[
                "incident_category"
            ]
            .value_counts()
            .to_dict()
        )

        composition[
            str(cluster_id)
        ] = {
            "size":
                int(
                    len(
                        cluster_rows
                    )
                ),

            "categories":
                {
                    str(key):
                        int(value)

                    for key, value
                    in category_counts.items()
                },
        }

    return composition


# =========================================================
# RUN CLUSTERING EXPERIMENT
# =========================================================

def evaluate_clusters(
    matrix,
    dataframe,
):

    print()
    print("=" * 70)
    print("CLUSTERING EVALUATION")
    print("=" * 70)

    results = []

    for k in CLUSTER_COUNTS:

        print()
        print("-" * 70)

        print(
            f"Testing K = {k}"
        )

        # -------------------------------------------------
        # K-MEANS
        # -------------------------------------------------

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        cluster_labels = (
            model.fit_predict(
                matrix
            )
        )

        # -------------------------------------------------
        # SILHOUETTE
        # -------------------------------------------------

        silhouette = silhouette_score(
            matrix,
            cluster_labels,
            metric="euclidean",
        )

        # -------------------------------------------------
        # DAVIES-BOULDIN
        # -------------------------------------------------

        dbi = davies_bouldin_score(
            matrix,
            cluster_labels,
        )

        # -------------------------------------------------
        # CALINSKI-HARABASZ
        # -------------------------------------------------

        chi = calinski_harabasz_score(
            matrix,
            cluster_labels,
        )

        # -------------------------------------------------
        # CLUSTER COMPOSITION
        # -------------------------------------------------

        composition = (
            calculate_cluster_composition(
                dataframe,
                cluster_labels,
                k,
            )
        )

        result = {
            "k":
                k,

            "silhouette_score":
                round(
                    float(
                        silhouette
                    ),
                    4,
                ),

            "davies_bouldin_index":
                round(
                    float(
                        dbi
                    ),
                    4,
                ),

            "calinski_harabasz_index":
                round(
                    float(
                        chi
                    ),
                    4,
                ),

            "cluster_composition":
                composition,
        }

        results.append(
            result
        )

        print(
            f"Silhouette Score       : "
            f"{silhouette:.4f}"
        )

        print(
            f"Davies-Bouldin Index   : "
            f"{dbi:.4f}"
        )

        print(
            f"Calinski-Harabasz Index: "
            f"{chi:.4f}"
        )

    return results


# =========================================================
# IDENTIFY BEST RESULTS
# =========================================================

def identify_best_results(
    results
):

    best_silhouette = max(
        results,
        key=lambda item:
            item[
                "silhouette_score"
            ],
    )

    best_dbi = min(
        results,
        key=lambda item:
            item[
                "davies_bouldin_index"
            ],
    )

    best_chi = max(
        results,
        key=lambda item:
            item[
                "calinski_harabasz_index"
            ],
    )

    print()
    print("=" * 70)
    print("BEST CLUSTERING RESULTS")
    print("=" * 70)

    print()
    print(
        "Best Silhouette Score:"
    )

    print(
        f"K = {best_silhouette['k']} "
        f"| Score = "
        f"{best_silhouette['silhouette_score']}"
    )

    print()
    print(
        "Best Davies-Bouldin Index:"
    )

    print(
        f"K = {best_dbi['k']} "
        f"| DBI = "
        f"{best_dbi['davies_bouldin_index']}"
    )

    print()
    print(
        "Best Calinski-Harabasz Index:"
    )

    print(
        f"K = {best_chi['k']} "
        f"| CHI = "
        f"{best_chi['calinski_harabasz_index']}"
    )

    return {
        "best_silhouette": {
            "k":
                best_silhouette[
                    "k"
                ],

            "score":
                best_silhouette[
                    "silhouette_score"
                ],
        },

        "best_davies_bouldin": {
            "k":
                best_dbi[
                    "k"
                ],

            "score":
                best_dbi[
                    "davies_bouldin_index"
                ],
        },

        "best_calinski_harabasz": {
            "k":
                best_chi[
                    "k"
                ],

            "score":
                best_chi[
                    "calinski_harabasz_index"
                ],
        },
    }


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(
    results,
    best_results,
    explained_variance,
    sample_size,
):

    output = {
        "experiment":
            "TF-IDF clustering evaluation",

        "reference_source":
            "80% training/reference set",

        "retrieval_configuration":
            "V3_with_description",

        "sample_size":
            sample_size,

        "samples_per_category":
            SAMPLES_PER_CATEGORY,

        "svd_components":
            SVD_COMPONENTS,

        "svd_explained_variance":
            round(
                float(
                    explained_variance
                ),
                4,
            ),

        "random_state":
            RANDOM_STATE,

        "cluster_counts_tested":
            CLUSTER_COUNTS,

        "results":
            results,

        "best_results":
            best_results,
    }

    with open(
        RESULT_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    comparison_rows = []

    for result in results:

        comparison_rows.append(
            {
                "k":
                    result[
                        "k"
                    ],

                "silhouette_score":
                    result[
                        "silhouette_score"
                    ],

                "davies_bouldin_index":
                    result[
                        "davies_bouldin_index"
                    ],

                "calinski_harabasz_index":
                    result[
                        "calinski_harabasz_index"
                    ],
            }
        )

    comparison_dataframe = (
        pd.DataFrame(
            comparison_rows
        )
    )

    comparison_dataframe.to_csv(
        RESULT_CSV_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("CLUSTERING EVALUATION COMPLETE")
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
        "CSV comparison saved to:"
    )

    print(
        RESULT_CSV_FILE
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("RAG CLUSTERING EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load balanced subset
    # -----------------------------------------------------

    dataframe = (
        load_balanced_sample()
    )

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    tfidf_matrix = (
        build_tfidf_matrix(
            dataframe
        )
    )

    # -----------------------------------------------------
    # SVD
    # -----------------------------------------------------

    (
        reduced_matrix,
        explained_variance,
    ) = reduce_dimensions(
        tfidf_matrix
    )

    # -----------------------------------------------------
    # Evaluate clustering
    # -----------------------------------------------------

    results = (
        evaluate_clusters(
            reduced_matrix,
            dataframe,
        )
    )

    # -----------------------------------------------------
    # Find best metrics
    # -----------------------------------------------------

    best_results = (
        identify_best_results(
            results
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_results(
        results,
        best_results,
        explained_variance,
        len(
            dataframe
        ),
    )


if __name__ == "__main__":

    main()