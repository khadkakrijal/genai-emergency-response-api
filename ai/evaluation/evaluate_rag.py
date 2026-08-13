import json
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ai.classifier import EmergencyClassifier


classifier = EmergencyClassifier()

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_CASES_FILE = os.path.join(
    CURRENT_DIR,
    "test_cases.json",
)

OUTPUT_FILE = os.path.join(
    CURRENT_DIR,
    "evaluation_results.json",
)


# =========================================================
# NORMALISATION
# =========================================================

def normalise_incident_type(value):

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if any(term in value for term in [
        "armed threat",
        "assault with a weapon",
        "threat with a weapon",
        "armed assault",
        "weapon assault",
        "assault victim",
        "assault with weapon",
        "weapon threat",
    ]):
        return "armed_threat"

    if any(term in value for term in [
        "traffic",
        "traffic accident",
        "vehicle accident",
        "vehicle collision",
        "car accident",
        "road accident",
        "traffic collision",
    ]):
        return "traffic_accident"

    if any(term in value for term in [
        "fire emergency",
        "building fire",
        "house fire",
        "residential fire",
        "structure fire",
        "fire incident",
    ]):
        return "fire_emergency"

    if any(term in value for term in [
        "medical",
        "medical emergency",
        "medical incident",
        "ems emergency",
        "health emergency",
    ]):
        return "medical_emergency"

    if value in [
        "uncertain",
        "unknown",
        "insufficient evidence",
        "human review required",
    ]:
        return "uncertain"

    return value.replace(" ", "_")


def normalise_risk_level(value):

    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value in [
        "critical",
        "severe",
        "high",
        "high risk",
    ]:
        return "high"

    if value in [
        "moderate",
        "medium",
        "medium risk",
    ]:
        return "medium"

    if value in [
        "minor",
        "low",
        "low risk",
    ]:
        return "low"

    if value in [
        "uncertain",
        "unknown",
        "insufficient evidence",
    ]:
        return "uncertain"

    return value


# =========================================================
# RETRIEVAL EVALUATION
# =========================================================

def is_retrieval_relevant(
    incident,
    relevant_keywords,
):
    """
    Determine whether a retrieved historical incident
    is relevant based on expected title/category keywords.
    """

    if not relevant_keywords:
        return False

    title = str(
        incident.get("title") or ""
    ).upper()

    incident_type = str(
        incident.get("incident_type") or ""
    ).upper()

    searchable_text = (
        f"{title} {incident_type}"
    )

    for keyword in relevant_keywords:

        if str(keyword).upper() in searchable_text:
            return True

    return False


def calculate_precision_at_k(
    retrieved_incidents,
    relevant_keywords,
    k=5,
):
    """
    Precision@K =
    relevant retrieved items in top K / K

    If fewer than K incidents are returned,
    denominator uses the number actually returned.
    """

    top_results = retrieved_incidents[:k]

    if not top_results:
        return 0.0, 0

    relevant_count = sum(
        1
        for incident in top_results
        if is_retrieval_relevant(
            incident,
            relevant_keywords,
        )
    )

    precision_at_k = (
        relevant_count
        /
        len(top_results)
    )

    return (
        round(precision_at_k, 4),
        relevant_count,
    )


# =========================================================
# LOAD TEST CASES
# =========================================================

def load_test_cases():

    with open(
        TEST_CASES_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# RUN EVALUATION
# =========================================================

def run_evaluation():

    test_cases = load_test_cases()

    classification_expected_types = []
    classification_predicted_types = []

    classification_expected_risks = []
    classification_predicted_risks = []

    precision_at_5_scores = []

    guardrail_tests = 0
    guardrail_correct = 0

    evaluation_results = []

    print()
    print("=" * 70)
    print("RAG PILOT EVALUATION")
    print("=" * 70)

    print(
        f"Number of test cases: {len(test_cases)}"
    )

    # =====================================================
    # TEST EACH INCIDENT
    # =====================================================

    for test_case in test_cases:

        print()
        print("-" * 70)

        test_id = test_case["id"]

        evaluation_mode = test_case.get(
            "evaluation_mode",
            "classification",
        )

        print(
            f"Test ID: {test_id}"
        )

        print(
            f"Evaluation Mode: {evaluation_mode}"
        )

        print(
            f"Incident: {test_case['description']}"
        )

        incident_data = {
            "description":
                test_case["description"],

            "location":
                test_case.get("location"),

            "incident_time":
                test_case.get("incident_time"),

            "people_involved":
                test_case.get("people_involved"),

            "weapon_involved":
                test_case.get("weapon_involved"),

            "injury_reported":
                test_case.get("injury_reported"),

            "location_type":
                test_case.get("location_type"),
        }

        try:

            result = classifier.classify(
                incident_data
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

            result = {
                "incident_type": "Error",
                "risk_level": "Error",
                "needs_human_review": True,
                "highest_similarity_score": 0,
                "similar_incidents": [],
            }

        # =================================================
        # EXPECTED VALUES
        # =================================================

        expected_type_raw = test_case[
            "expected_incident_type"
        ]

        expected_risk_raw = test_case[
            "expected_risk_level"
        ]

        expected_human_review = test_case.get(
            "expected_human_review",
            False,
        )

        relevant_keywords = test_case.get(
            "relevant_retrieval_keywords",
            [],
        )

        # =================================================
        # ACTUAL VALUES
        # =================================================

        predicted_type_raw = result.get(
            "incident_type",
            "Unknown",
        )

        predicted_risk_raw = result.get(
            "risk_level",
            "Unknown",
        )

        predicted_human_review = result.get(
            "needs_human_review",
            False,
        )

        similar_incidents = result.get(
            "similar_incidents",
            [],
        )

        highest_similarity = result.get(
            "highest_similarity_score",
            0,
        )

        # =================================================
        # NORMALISE
        # =================================================

        expected_type = normalise_incident_type(
            expected_type_raw
        )

        predicted_type = normalise_incident_type(
            predicted_type_raw
        )

        expected_risk = normalise_risk_level(
            expected_risk_raw
        )

        predicted_risk = normalise_risk_level(
            predicted_risk_raw
        )

        # =================================================
        # CORRECTNESS
        # =================================================

        type_correct = (
            expected_type
            ==
            predicted_type
        )

        risk_correct = (
            expected_risk
            ==
            predicted_risk
        )

        human_review_correct = (
            expected_human_review
            ==
            predicted_human_review
        )

        # =================================================
        # PRECISION@5
        # =================================================

        precision_at_5 = None
        relevant_in_top_5 = None

        if evaluation_mode == "classification":

            (
                precision_at_5,
                relevant_in_top_5,
            ) = calculate_precision_at_k(
                retrieved_incidents=
                    similar_incidents,

                relevant_keywords=
                    relevant_keywords,

                k=5,
            )

            precision_at_5_scores.append(
                precision_at_5
            )

        # =================================================
        # CLASSIFICATION METRICS
        # =================================================

        if evaluation_mode == "classification":

            classification_expected_types.append(
                expected_type
            )

            classification_predicted_types.append(
                predicted_type
            )

            classification_expected_risks.append(
                expected_risk
            )

            classification_predicted_risks.append(
                predicted_risk
            )

        # =================================================
        # GUARDRAIL METRICS
        # =================================================

        elif evaluation_mode == "guardrail":

            guardrail_tests += 1

            if human_review_correct:
                guardrail_correct += 1

        # =================================================
        # PRINT
        # =================================================

        print()

        print(
            f"Expected Type : "
            f"{expected_type_raw}"
        )

        print(
            f"Predicted Type: "
            f"{predicted_type_raw}"
        )

        print()

        print(
            f"Expected Risk : "
            f"{expected_risk_raw}"
        )

        print(
            f"Predicted Risk: "
            f"{predicted_risk_raw}"
        )

        print()

        print(
            f"Type Correct  : "
            f"{type_correct}"
        )

        print(
            f"Risk Correct  : "
            f"{risk_correct}"
        )

        print()

        print(
            f"Expected Human Review: "
            f"{expected_human_review}"
        )

        print(
            f"Actual Human Review  : "
            f"{predicted_human_review}"
        )

        print(
            f"Human Review Correct : "
            f"{human_review_correct}"
        )

        print()

        print(
            f"Highest Similarity: "
            f"{highest_similarity}"
        )

        if precision_at_5 is not None:

            print(
                f"Relevant in Top 5: "
                f"{relevant_in_top_5}/5"
            )

            print(
                f"Precision@5: "
                f"{precision_at_5:.3f}"
            )

        print()

        print(
            "Top Historical Incidents:"
        )

        if not similar_incidents:

            print(
                "No historical incidents retrieved."
            )

        else:

            for index, incident in enumerate(
                similar_incidents,
                start=1,
            ):

                relevant = (
                    is_retrieval_relevant(
                        incident,
                        relevant_keywords,
                    )
                    if relevant_keywords
                    else False
                )

                relevance_label = (
                    "Relevant"
                    if relevant
                    else "Not Relevant"
                )

                print(
                    f"{index}. "
                    f"{incident.get('title', 'Unknown')} "
                    f"| Score: "
                    f"{incident.get('similarity_score', 0)} "
                    f"| {relevance_label}"
                )

        # =================================================
        # SAVE INDIVIDUAL RESULT
        # =================================================

        evaluation_results.append(
            {
                "test_id":
                    test_id,

                "evaluation_mode":
                    evaluation_mode,

                "description":
                    test_case["description"],

                "expected_incident_type":
                    expected_type_raw,

                "predicted_incident_type":
                    predicted_type_raw,

                "incident_type_correct":
                    type_correct,

                "expected_risk_level":
                    expected_risk_raw,

                "predicted_risk_level":
                    predicted_risk_raw,

                "risk_level_correct":
                    risk_correct,

                "expected_human_review":
                    expected_human_review,

                "actual_human_review":
                    predicted_human_review,

                "human_review_correct":
                    human_review_correct,

                "highest_similarity_score":
                    highest_similarity,

                "precision_at_5":
                    precision_at_5,

                "relevant_in_top_5":
                    relevant_in_top_5,

                "retrieved_incidents":
                    similar_incidents,
            }
        )

    # =====================================================
    # INCIDENT TYPE METRICS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "INCIDENT TYPE PERFORMANCE"
    )
    print("=" * 70)

    type_accuracy = accuracy_score(
        classification_expected_types,
        classification_predicted_types,
    )

    type_precision = precision_score(
        classification_expected_types,
        classification_predicted_types,
        average="weighted",
        zero_division=0,
    )

    type_recall = recall_score(
        classification_expected_types,
        classification_predicted_types,
        average="weighted",
        zero_division=0,
    )

    type_f1 = f1_score(
        classification_expected_types,
        classification_predicted_types,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Accuracy : {type_accuracy:.3f}"
    )

    print(
        f"Precision: {type_precision:.3f}"
    )

    print(
        f"Recall   : {type_recall:.3f}"
    )

    print(
        f"F1 Score : {type_f1:.3f}"
    )

    # =====================================================
    # RISK LEVEL METRICS
    # =====================================================

    print()
    print("=" * 70)
    print(
        "RISK LEVEL PERFORMANCE"
    )
    print("=" * 70)

    risk_accuracy = accuracy_score(
        classification_expected_risks,
        classification_predicted_risks,
    )

    risk_precision = precision_score(
        classification_expected_risks,
        classification_predicted_risks,
        average="weighted",
        zero_division=0,
    )

    risk_recall = recall_score(
        classification_expected_risks,
        classification_predicted_risks,
        average="weighted",
        zero_division=0,
    )

    risk_f1 = f1_score(
        classification_expected_risks,
        classification_predicted_risks,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Accuracy : {risk_accuracy:.3f}"
    )

    print(
        f"Precision: {risk_precision:.3f}"
    )

    print(
        f"Recall   : {risk_recall:.3f}"
    )

    print(
        f"F1 Score : {risk_f1:.3f}"
    )

    # =====================================================
    # RETRIEVAL PERFORMANCE
    # =====================================================

    print()
    print("=" * 70)
    print(
        "RAG RETRIEVAL PERFORMANCE"
    )
    print("=" * 70)

    average_precision_at_5 = (
        sum(precision_at_5_scores)
        /
        len(precision_at_5_scores)
        if precision_at_5_scores
        else 0
    )

    print(
        f"Classification Cases: "
        f"{len(precision_at_5_scores)}"
    )

    print(
        f"Mean Precision@5    : "
        f"{average_precision_at_5:.3f}"
    )

    # =====================================================
    # GUARDRAIL PERFORMANCE
    # =====================================================

    print()
    print("=" * 70)
    print(
        "GUARDRAIL PERFORMANCE"
    )
    print("=" * 70)

    guardrail_accuracy = (
        guardrail_correct
        /
        guardrail_tests
        if guardrail_tests > 0
        else 0
    )

    print(
        f"Guardrail Tests   : "
        f"{guardrail_tests}"
    )

    print(
        f"Correct Decisions : "
        f"{guardrail_correct}"
    )

    print(
        f"Guardrail Accuracy: "
        f"{guardrail_accuracy:.3f}"
    )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================

    summary = {
        "evaluation_type":
            "pilot",

        "total_test_cases":
            len(test_cases),

        "classification_test_cases":
            len(
                classification_expected_types
            ),

        "guardrail_test_cases":
            guardrail_tests,

        "incident_type_metrics": {
            "accuracy":
                round(type_accuracy, 4),

            "precision":
                round(type_precision, 4),

            "recall":
                round(type_recall, 4),

            "f1_score":
                round(type_f1, 4),
        },

        "risk_level_metrics": {
            "accuracy":
                round(risk_accuracy, 4),

            "precision":
                round(risk_precision, 4),

            "recall":
                round(risk_recall, 4),

            "f1_score":
                round(risk_f1, 4),
        },

        "retrieval_metrics": {
            "mean_precision_at_5":
                round(
                    average_precision_at_5,
                    4,
                ),

            "classification_cases":
                len(
                    precision_at_5_scores
                ),
        },

        "guardrail_metrics": {
            "tests":
                guardrail_tests,

            "correct":
                guardrail_correct,

            "accuracy":
                round(
                    guardrail_accuracy,
                    4,
                ),
        },

        "results":
            evaluation_results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_evaluation()