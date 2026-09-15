import os
import json
import pandas as pd


# =========================================================
# PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HOLDOUT_FILE = os.path.join(
    CURRENT_DIR,
    "holdout_evaluation_results.json",
)

RETRIEVER_FILE = os.path.join(
    CURRENT_DIR,
    "retriever_comparison_results.json",
)

THRESHOLD_FILE = os.path.join(
    CURRENT_DIR,
    "threshold_evaluation_results.json",
)

CLUSTERING_FILE = os.path.join(
    CURRENT_DIR,
    "clustering_evaluation_results.json",
)

LLM_PROMPT_FILE = os.path.join(
    CURRENT_DIR,
    "llm_prompt_evaluation_results.json",
)

HALLUCINATION_FILE = os.path.join(
    CURRENT_DIR,
    "hallucination_evaluation_results.json",
)


OUTPUT_JSON = os.path.join(
    CURRENT_DIR,
    "final_evaluation_summary.json",
)

OUTPUT_CSV = os.path.join(
    CURRENT_DIR,
    "final_evaluation_summary.csv",
)


# =========================================================
# LOAD JSON
# =========================================================

def load_json(file_path, label):

    if not os.path.exists(file_path):

        print(
            f"WARNING: {label} file not found:"
        )

        print(file_path)

        return None

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# HELPER
# =========================================================

def safe_get(
    data,
    *keys,
    default=None,
):
    """
    Safely retrieve nested dictionary values.
    """

    current = data

    for key in keys:

        if not isinstance(
            current,
            dict,
        ):
            return default

        current = current.get(
            key
        )

        if current is None:
            return default

    return current


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("GENERATING FINAL EVALUATION SUMMARY")
    print("=" * 70)

    # -----------------------------------------------------
    # Load experiment results
    # -----------------------------------------------------

    holdout = load_json(
        HOLDOUT_FILE,
        "Holdout evaluation",
    )

    retriever = load_json(
        RETRIEVER_FILE,
        "Retriever comparison",
    )

    threshold = load_json(
        THRESHOLD_FILE,
        "Threshold evaluation",
    )

    clustering = load_json(
        CLUSTERING_FILE,
        "Clustering evaluation",
    )

    llm_prompt = load_json(
        LLM_PROMPT_FILE,
        "LLM prompt evaluation",
    )

    hallucination = load_json(
        HALLUCINATION_FILE,
        "Hallucination evaluation",
    )

    # =====================================================
    # 1. DATASET SPLIT
    # =====================================================

    dataset_summary = {
        "full_dataset_records":
            663522,

        "reference_set_records":
            530817,

        "test_set_records":
            132705,

        "reference_percentage":
            80,

        "test_percentage":
            20,

        "split_method":
            "Stratified random split",

        "random_state":
            42,
    }

    # =====================================================
    # 2. BASELINE HOLDOUT RESULTS
    # =====================================================

    holdout_summary = {}

    if holdout:

        holdout_summary = {
            "sample_size":
                safe_get(
                    holdout,
                    "evaluation",
                    "sample_size",
                ),

            "samples_per_category":
                safe_get(
                    holdout,
                    "evaluation",
                    "samples_per_category",
                ),

            "accuracy":
                safe_get(
                    holdout,
                    "classification_metrics",
                    "accuracy",
                ),

            "precision":
                safe_get(
                    holdout,
                    "classification_metrics",
                    "precision",
                ),

            "recall":
                safe_get(
                    holdout,
                    "classification_metrics",
                    "recall",
                ),

            "f1_score":
                safe_get(
                    holdout,
                    "classification_metrics",
                    "f1_score",
                ),

            "mean_precision_at_5":
                safe_get(
                    holdout,
                    "retrieval_metrics",
                    "mean_precision_at_5",
                ),
        }

    # =====================================================
    # 3. RETRIEVER COMPARISON
    # =====================================================

    retriever_summary = {}

    if retriever:

        best_retriever = retriever.get(
            "best_configuration"
        )

        retriever_results = retriever.get(
            "results",
            {},
        )

        retriever_summary = {
            "best_configuration":
                best_retriever,

            "best_metrics":
                retriever_results.get(
                    best_retriever,
                    {},
                ),

            "all_configurations":
                retriever_results,
        }

    # =====================================================
    # 4. THRESHOLD EVALUATION
    # =====================================================

    threshold_summary = {}

    if threshold:

        threshold_summary = {
            "retriever_configuration":
                threshold.get(
                    "retriever_configuration"
                ),

            "thresholds_tested":
                threshold.get(
                    "thresholds_tested"
                ),

            "selected_threshold":
                threshold.get(
                    "best_threshold"
                ),

            "selected_threshold_metrics":
                threshold.get(
                    "best_threshold_summary"
                ),

            "all_threshold_results":
                threshold.get(
                    "threshold_results"
                ),
        }

    # =====================================================
    # 5. CLUSTERING
    # =====================================================

    clustering_summary = {}

    if clustering:

        clustering_summary = {
            "sample_size":
                clustering.get(
                    "sample_size"
                ),

            "samples_per_category":
                clustering.get(
                    "samples_per_category"
                ),

            "svd_components":
                clustering.get(
                    "svd_components"
                ),

            "svd_explained_variance":
                clustering.get(
                    "svd_explained_variance"
                ),

            "best_silhouette":
                safe_get(
                    clustering,
                    "best_results",
                    "best_silhouette",
                ),

            "best_davies_bouldin":
                safe_get(
                    clustering,
                    "best_results",
                    "best_davies_bouldin",
                ),

            "best_calinski_harabasz":
                safe_get(
                    clustering,
                    "best_results",
                    "best_calinski_harabasz",
                ),

            "all_results":
                clustering.get(
                    "results"
                ),
        }

    # =====================================================
    # 6. LLM PROMPT COMPARISON
    # =====================================================

    llm_summary = {}

    if llm_prompt:

        llm_summary = {
            "test_cases":
                llm_prompt.get(
                    "test_cases"
                ),

            "best_prompt":
                llm_prompt.get(
                    "best_prompt"
                ),

            "prompt_metrics":
                llm_prompt.get(
                    "metrics"
                ),

            "prompt_descriptions":
                llm_prompt.get(
                    "prompts"
                ),
        }

    # =====================================================
    # 7. HALLUCINATION
    # =====================================================

    hallucination_summary = {}

    if hallucination:

        hallucination_summary = {
            "test_cases":
                hallucination.get(
                    "test_cases"
                ),

            "best_safety_prompt":
                hallucination.get(
                    "best_safety_prompt"
                ),

            "prompt_metrics":
                hallucination.get(
                    "metrics"
                ),
        }

    # =====================================================
    # FINAL COMBINED SUMMARY
    # =====================================================

    final_summary = {
        "project":
            "Advanced Situational Awareness and Intelligence "
            "for First Responders using Artificial Intelligence",

        "evaluation_overview": {
            "dataset_split":
                dataset_summary,

            "baseline_holdout_evaluation":
                holdout_summary,

            "retriever_comparison":
                retriever_summary,

            "threshold_evaluation":
                threshold_summary,

            "clustering_evaluation":
                clustering_summary,

            "llm_prompt_evaluation":
                llm_summary,

            "hallucination_evaluation":
                hallucination_summary,
        },

        "key_findings": {
            "baseline_accuracy":
                holdout_summary.get(
                    "accuracy"
                ),

            "baseline_f1":
                holdout_summary.get(
                    "f1_score"
                ),

            "best_retriever":
                retriever_summary.get(
                    "best_configuration"
                ),

            "best_retriever_accuracy":
                safe_get(
                    retriever_summary,
                    "best_metrics",
                    "accuracy",
                ),

            "best_retriever_f1":
                safe_get(
                    retriever_summary,
                    "best_metrics",
                    "f1_score",
                ),

            "selected_similarity_threshold":
                threshold_summary.get(
                    "selected_threshold"
                ),

            "best_clustering_silhouette":
                clustering_summary.get(
                    "best_silhouette"
                ),

            "best_clustering_dbi":
                clustering_summary.get(
                    "best_davies_bouldin"
                ),

            "best_clustering_chi":
                clustering_summary.get(
                    "best_calinski_harabasz"
                ),

            "best_classification_prompt":
                llm_summary.get(
                    "best_prompt"
                ),

            "best_safety_prompt":
                hallucination_summary.get(
                    "best_safety_prompt"
                ),
        },
    }

    # =====================================================
    # SAVE JSON
    # =====================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            indent=4,
        )

    # =====================================================
    # CREATE REPORT-FRIENDLY CSV
    # =====================================================

    rows = []

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    rows.append(
        {
            "Experiment":
                "Dataset Split",

            "Metric":
                "Reference Set",

            "Value":
                dataset_summary[
                    "reference_set_records"
                ],
        }
    )

    rows.append(
        {
            "Experiment":
                "Dataset Split",

            "Metric":
                "Test Set",

            "Value":
                dataset_summary[
                    "test_set_records"
                ],
        }
    )

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    if holdout_summary:

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mean_precision_at_5",
        ]:

            rows.append(
                {
                    "Experiment":
                        "Baseline Holdout",

                    "Metric":
                        metric,

                    "Value":
                        holdout_summary.get(
                            metric
                        ),
                }
            )

    # -----------------------------------------------------
    # Best Retriever
    # -----------------------------------------------------

    if retriever_summary:

        best_name = (
            retriever_summary.get(
                "best_configuration"
            )
        )

        best_metrics = (
            retriever_summary.get(
                "best_metrics",
                {},
            )
        )

        for metric, value in (
            best_metrics.items()
        ):

            rows.append(
                {
                    "Experiment":
                        f"Best Retriever - {best_name}",

                    "Metric":
                        metric,

                    "Value":
                        value,
                }
            )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    if threshold_summary:

        selected_metrics = (
            threshold_summary.get(
                "selected_threshold_metrics",
                {},
            )
            or {}
        )

        rows.append(
            {
                "Experiment":
                    "Threshold Evaluation",

                "Metric":
                    "selected_threshold",

                "Value":
                    threshold_summary.get(
                        "selected_threshold"
                    ),
            }
        )

        for metric, value in (
            selected_metrics.items()
        ):

            rows.append(
                {
                    "Experiment":
                        "Threshold Evaluation",

                    "Metric":
                        metric,

                    "Value":
                        value,
                }
            )

    # -----------------------------------------------------
    # Clustering
    # -----------------------------------------------------

    if clustering_summary:

        rows.append(
            {
                "Experiment":
                    "Clustering",

                "Metric":
                    "SVD explained variance",

                "Value":
                    clustering_summary.get(
                        "svd_explained_variance"
                    ),
            }
        )

        rows.append(
            {
                "Experiment":
                    "Clustering",

                "Metric":
                    "Best Silhouette",

                "Value":
                    clustering_summary.get(
                        "best_silhouette"
                    ),
            }
        )

        rows.append(
            {
                "Experiment":
                    "Clustering",

                "Metric":
                    "Best DBI",

                "Value":
                    clustering_summary.get(
                        "best_davies_bouldin"
                    ),
            }
        )

        rows.append(
            {
                "Experiment":
                    "Clustering",

                "Metric":
                    "Best CHI",

                "Value":
                    clustering_summary.get(
                        "best_calinski_harabasz"
                    ),
            }
        )

    # -----------------------------------------------------
    # LLM Prompt Comparison
    # -----------------------------------------------------

    if llm_summary:

        prompt_metrics = (
            llm_summary.get(
                "prompt_metrics",
                {},
            )
            or {}
        )

        for prompt_name, metrics in (
            prompt_metrics.items()
        ):

            for metric, value in (
                metrics.items()
            ):

                rows.append(
                    {
                        "Experiment":
                            f"LLM Prompt - {prompt_name}",

                        "Metric":
                            metric,

                        "Value":
                            value,
                    }
                )

    # -----------------------------------------------------
    # Hallucination
    # -----------------------------------------------------

    if hallucination_summary:

        hallucination_metrics = (
            hallucination_summary.get(
                "prompt_metrics",
                {},
            )
            or {}
        )

        for prompt_name, metrics in (
            hallucination_metrics.items()
        ):

            for metric, value in (
                metrics.items()
            ):

                rows.append(
                    {
                        "Experiment":
                            f"Hallucination - {prompt_name}",

                        "Metric":
                            metric,

                        "Value":
                            value,
                    }
                )

    summary_dataframe = pd.DataFrame(
        rows
    )

    summary_dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =====================================================
    # PRINT MAIN FINDINGS
    # =====================================================

    print()
    print("=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    print()

    print(
        f"Baseline Accuracy: "
        f"{final_summary['key_findings']['baseline_accuracy']}"
    )

    print(
        f"Baseline F1: "
        f"{final_summary['key_findings']['baseline_f1']}"
    )

    print()

    print(
        f"Best Retriever: "
        f"{final_summary['key_findings']['best_retriever']}"
    )

    print(
        f"Best Retriever Accuracy: "
        f"{final_summary['key_findings']['best_retriever_accuracy']}"
    )

    print(
        f"Best Retriever F1: "
        f"{final_summary['key_findings']['best_retriever_f1']}"
    )

    print()

    print(
        f"Selected Threshold: "
        f"{final_summary['key_findings']['selected_similarity_threshold']}"
    )

    print()

    print(
        f"Best Classification Prompt: "
        f"{final_summary['key_findings']['best_classification_prompt']}"
    )

    print(
        f"Best Safety Prompt: "
        f"{final_summary['key_findings']['best_safety_prompt']}"
    )

    print()
    print("=" * 70)
    print("SUMMARY GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("JSON summary:")
    print(OUTPUT_JSON)

    print()
    print("CSV summary:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()