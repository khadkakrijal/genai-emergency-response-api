import os
import json
import re
import time

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ai.evaluation.reference_retriever import ReferenceRetriever
from ai.ollama_service import generate_with_ollama
from ai.parser import parse_ai_response


# =========================================================
# PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_FILE = os.path.join(
    CURRENT_DIR,
    "test_set.csv",
)

RESULT_FILE = os.path.join(
    CURRENT_DIR,
    "llm_prompt_evaluation_results.json",
)

PREDICTION_FILE = os.path.join(
    CURRENT_DIR,
    "llm_prompt_predictions.csv",
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# Start small because every test is sent to Llama 3 times.
SAMPLES_PER_CATEGORY = 10

TOP_K = 5

PROMPT_TYPES = [
    "A_BASIC",
    "B_RAG",
    "C_RAG_GUARDRAIL",
]


# =========================================================
# CATEGORY NORMALISATION
# =========================================================

def normalise_category(value):

    if value is None:
        return "Unknown"

    value = str(value).strip().lower()

    # EMS / Medical
    if any(
        term in value
        for term in [
            "ems",
            "medical",
            "medical emergency",
            "health emergency",
            "cardiac",
            "respiratory",
            "dizziness",
        ]
    ):
        return "EMS"

    # Traffic
    if any(
        term in value
        for term in [
            "traffic",
            "vehicle accident",
            "vehicle collision",
            "car accident",
            "road accident",
            "collision",
        ]
    ):
        return "Traffic"

    # Fire
    if any(
        term in value
        for term in [
            "fire",
            "building fire",
            "vehicle fire",
            "fire alarm",
            "burn",
            "gas leak",
        ]
    ):
        return "Fire"

    if any(
        term in value
        for term in [
            "uncertain",
            "unknown",
        ]
    ):
        return "Uncertain"

    return value


# =========================================================
# BUILD TEST INPUT
# =========================================================

def build_incident_text(row):
    """
    Remove the ground-truth prefix so Llama is not directly
    told that the incident is EMS, Traffic or Fire.

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

    return (
        f"{title}\n"
        f"Additional details: {description}"
    ).strip()


# =========================================================
# LOAD BALANCED HELD-OUT SAMPLE
# =========================================================

def load_test_sample():

    print()
    print("=" * 70)
    print("LOADING HELD-OUT LLM TEST SAMPLE")
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

        samples.append(sample)

    test_sample = pd.concat(
        samples,
        ignore_index=True,
    )

    test_sample = (
        test_sample
        .sample(
            frac=1,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"Total LLM evaluation cases: "
        f"{len(test_sample)}"
    )

    print()

    print(
        test_sample[
            "incident_category"
        ].value_counts()
    )

    return test_sample


# =========================================================
# FORMAT RAG CONTEXT
# =========================================================

def format_historical_incidents(
    similar_incidents
):

    if not similar_incidents:
        return "No relevant historical incidents retrieved."

    lines = []

    for index, incident in enumerate(
        similar_incidents,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"Title: {incident.get('title')} | "
            f"Category: {incident.get('incident_type')} | "
            f"Similarity: {incident.get('similarity_score')}"
        )

    return "\n".join(lines)


# =========================================================
# PROMPT A
# BASIC LLM
# =========================================================

def build_prompt_a(
    incident_text
):

    return f"""
You are an emergency incident classification assistant.

Analyse the following incident.

INCIDENT:

{incident_text}

Classify the incident into ONE of these categories:

EMS
Traffic
Fire

Return ONLY valid JSON:

{{
    "incident_type": "",
    "reasoning": ""
}}
"""


# =========================================================
# PROMPT B
# LLM + RAG
# =========================================================

def build_prompt_b(
    incident_text,
    historical_text,
):

    return f"""
You are an emergency incident classification assistant.

Analyse the current incident together with similar
historical emergency incidents.

CURRENT INCIDENT:

{incident_text}


SIMILAR HISTORICAL INCIDENTS:

{historical_text}


Use the historical incidents as supporting context.

Classify the current incident into ONE of:

EMS
Traffic
Fire

Return ONLY valid JSON:

{{
    "incident_type": "",
    "reasoning": ""
}}
"""


# =========================================================
# PROMPT C
# RAG + STRUCTURE + GUARDRAILS
# =========================================================

def build_prompt_c(
    incident_text,
    historical_text,
):

    return f"""
You are an AI decision-support assistant for emergency
first responders.

Analyse the current incident using the supplied incident
information and retrieved historical emergency incidents.

CURRENT INCIDENT:

{incident_text}


SIMILAR HISTORICAL INCIDENTS:

{historical_text}


CLASSIFICATION GUIDANCE:

EMS:
- Medical emergencies.
- Injury or illness.
- Cardiac emergencies.
- Breathing difficulties.
- Dizziness or weakness.
- Other incidents primarily requiring medical assistance.

TRAFFIC:
- Vehicle crashes.
- Traffic collisions.
- Disabled vehicles.
- Road incidents primarily involving transport or traffic.

FIRE:
- Building fires.
- Vehicle fires.
- Fire alarms.
- Smoke or active burning.
- Gas leaks.
- Fire service emergencies.


IMPORTANT SAFETY RULES:

- Historical incidents are supporting evidence only.
- Do not copy facts from a historical incident into the
  current incident.
- Do not invent injuries, victims, weapons, locations,
  events or other facts.
- Do not assume information that has not been provided.
- Base the classification on the current incident first,
  using the retrieved incidents only as supporting context.
- If the evidence is genuinely insufficient or conflicting,
  return "Uncertain".
- Do not provide a confident category when the available
  evidence does not support it.


Return ONLY valid JSON.

Do not include markdown or text outside the JSON.

{{
    "incident_type": "",
    "reasoning": ""
}}
"""


# =========================================================
# RUN ONE LLM PROMPT
# =========================================================

def run_llm_prompt(prompt):

    try:

        raw_response = generate_with_ollama(
            prompt
        )

        parsed_response = parse_ai_response(
            raw_response
        )

        if not isinstance(
            parsed_response,
            dict,
        ):
            return {
                "incident_type":
                    "Unknown",

                "reasoning":
                    "Invalid AI response.",
            }

        return parsed_response

    except Exception as error:

        print(
            f"\nLLM ERROR: {error}"
        )

        return {
            "incident_type":
                "Unknown",

            "reasoning":
                str(error),
        }


# =========================================================
# CALCULATE METRICS
# =========================================================

def calculate_metrics(
    expected_labels,
    predicted_labels,
):

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
    }


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():

    print()
    print("=" * 70)
    print("LLM PROMPT COMPARISON")
    print("=" * 70)

    # -----------------------------------------------------
    # Load unseen test cases
    # -----------------------------------------------------

    test_sample = load_test_sample()

    # -----------------------------------------------------
    # Initialise RAG reference retriever
    # -----------------------------------------------------

    retriever = ReferenceRetriever()

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    expected_labels = []

    predictions = {
        "A_BASIC": [],
        "B_RAG": [],
        "C_RAG_GUARDRAIL": [],
    }

    detailed_results = []

    total_cases = len(
        test_sample
    )

    # =====================================================
    # TEST EACH CASE
    # =====================================================

    for index, row in (
        test_sample.iterrows()
    ):

        test_number = (
            index + 1
        )

        expected_category = (
            normalise_category(
                row[
                    "incident_category"
                ]
            )
        )

        incident_text = (
            build_incident_text(
                row
            )
        )

        print()
        print("=" * 70)

        print(
            f"[{test_number}/{total_cases}]"
        )

        print(
            f"Expected: "
            f"{expected_category}"
        )

        print(
            f"Incident: "
            f"{row.get('title')}"
        )

        # -------------------------------------------------
        # RAG retrieval
        # -------------------------------------------------

        similar_incidents = (
            retriever.retrieve(
                description=
                    incident_text,

                top_k=TOP_K,
            )
        )

        historical_text = (
            format_historical_incidents(
                similar_incidents
            )
        )

        highest_similarity = (
            similar_incidents[0][
                "similarity_score"
            ]
            if similar_incidents
            else 0
        )

        # -------------------------------------------------
        # PROMPT A
        # -------------------------------------------------

        prompt_a = build_prompt_a(
            incident_text
        )

        result_a = run_llm_prompt(
            prompt_a
        )

        predicted_a = normalise_category(
            result_a.get(
                "incident_type"
            )
        )

        print(
            f"Prompt A → "
            f"{predicted_a}"
        )

        # -------------------------------------------------
        # PROMPT B
        # -------------------------------------------------

        prompt_b = build_prompt_b(
            incident_text,
            historical_text,
        )

        result_b = run_llm_prompt(
            prompt_b
        )

        predicted_b = normalise_category(
            result_b.get(
                "incident_type"
            )
        )

        print(
            f"Prompt B → "
            f"{predicted_b}"
        )

        # -------------------------------------------------
        # PROMPT C
        # -------------------------------------------------

        prompt_c = build_prompt_c(
            incident_text,
            historical_text,
        )

        result_c = run_llm_prompt(
            prompt_c
        )

        predicted_c = normalise_category(
            result_c.get(
                "incident_type"
            )
        )

        print(
            f"Prompt C → "
            f"{predicted_c}"
        )

        print(
            f"Highest RAG similarity: "
            f"{highest_similarity}"
        )

        # -------------------------------------------------
        # Store
        # -------------------------------------------------

        expected_labels.append(
            expected_category
        )

        predictions[
            "A_BASIC"
        ].append(
            predicted_a
        )

        predictions[
            "B_RAG"
        ].append(
            predicted_b
        )

        predictions[
            "C_RAG_GUARDRAIL"
        ].append(
            predicted_c
        )

        detailed_results.append(
            {
                "test_number":
                    test_number,

                "title":
                    row.get(
                        "title"
                    ),

                "expected_category":
                    expected_category,

                "prompt_a_prediction":
                    predicted_a,

                "prompt_b_prediction":
                    predicted_b,

                "prompt_c_prediction":
                    predicted_c,

                "prompt_a_correct":
                    predicted_a
                    ==
                    expected_category,

                "prompt_b_correct":
                    predicted_b
                    ==
                    expected_category,

                "prompt_c_correct":
                    predicted_c
                    ==
                    expected_category,

                "highest_similarity":
                    highest_similarity,

                "top_5_titles": [
                    incident.get(
                        "title"
                    )

                    for incident
                    in similar_incidents
                ],

                "top_5_categories": [
                    incident.get(
                        "incident_type"
                    )

                    for incident
                    in similar_incidents
                ],

                "prompt_a_reasoning":
                    result_a.get(
                        "reasoning"
                    ),

                "prompt_b_reasoning":
                    result_b.get(
                        "reasoning"
                    ),

                "prompt_c_reasoning":
                    result_c.get(
                        "reasoning"
                    ),
            }
        )

        # Small pause so local Ollama isn't hammered.
        time.sleep(
            0.2
        )

    # =====================================================
    # FINAL METRICS
    # =====================================================

    comparison = {}

    print()
    print("=" * 70)
    print("LLM PROMPT PERFORMANCE")
    print("=" * 70)

    for prompt_name in PROMPT_TYPES:

        metrics = calculate_metrics(
            expected_labels,
            predictions[
                prompt_name
            ],
        )

        comparison[
            prompt_name
        ] = metrics

        print()
        print(
            prompt_name
        )

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
            f"  F1 Score : "
            f"{metrics['f1_score']:.4f}"
        )

    # =====================================================
    # BEST PROMPT
    # =====================================================

    best_prompt = max(
        comparison,
        key=lambda prompt:
            comparison[
                prompt
            ][
                "f1_score"
            ],
    )

    print()
    print("=" * 70)

    print(
        f"BEST PROMPT: "
        f"{best_prompt}"
    )

    print("=" * 70)

    # =====================================================
    # SAVE JSON
    # =====================================================

    output = {
        "experiment":
            "LLM prompt comparison",

        "dataset":
            "20% held-out Kaggle 911 test set",

        "reference_source":
            "80% Kaggle 911 reference set",

        "test_cases":
            total_cases,

        "samples_per_category":
            SAMPLES_PER_CATEGORY,

        "top_k":
            TOP_K,

        "random_state":
            RANDOM_STATE,

        "prompts": {
            "A_BASIC":
                "Current incident only",

            "B_RAG":
                "Current incident plus RAG context",

            "C_RAG_GUARDRAIL":
                (
                    "Current incident plus RAG, "
                    "structured guidance and "
                    "anti-hallucination guardrails"
                ),
        },

        "metrics":
            comparison,

        "best_prompt":
            best_prompt,

        "results":
            detailed_results,
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

    # =====================================================
    # SAVE CSV
    # =====================================================

    pd.DataFrame(
        detailed_results
    ).to_csv(
        PREDICTION_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("LLM EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "JSON saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print(
        "CSV saved to:"
    )

    print(
        PREDICTION_FILE
    )


if __name__ == "__main__":

    main()