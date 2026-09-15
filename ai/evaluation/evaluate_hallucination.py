import os
import json
import re
import time

from ai.ollama_service import generate_with_ollama
from ai.parser import parse_ai_response

from ai.evaluation.reference_retriever import ReferenceRetriever


# =========================================================
# PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_FILE = os.path.join(
    CURRENT_DIR,
    "hallucination_test_cases.json",
)

RESULT_FILE = os.path.join(
    CURRENT_DIR,
    "hallucination_evaluation_results.json",
)


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K = 5

PROMPT_TYPES = [
    "A_BASIC",
    "B_RAG",
    "C_RAG_GUARDRAIL",
]


# =========================================================
# LOAD TEST CASES
# =========================================================

def load_test_cases():

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# FORMAT RAG CONTEXT
# =========================================================

def format_historical_incidents(
    similar_incidents
):

    if not similar_incidents:

        return (
            "No relevant historical incidents available."
        )

    lines = []

    for index, incident in enumerate(
        similar_incidents,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"{incident.get('title')} | "
            f"Category: {incident.get('incident_type')} | "
            f"Similarity: {incident.get('similarity_score')}"
        )

    return "\n".join(
        lines
    )


# =========================================================
# PROMPT A
# =========================================================

def build_prompt_a(
    description
):

    return f"""
You are an emergency response assistant.

Analyse this incident:

{description}

Return ONLY valid JSON:

{{
    "incident_type": "",
    "risk_level": "",
    "summary": "",
    "reasoning": "",
    "uncertain": false
}}
"""


# =========================================================
# PROMPT B
# =========================================================

def build_prompt_b(
    description,
    historical_text,
):

    return f"""
You are an emergency response assistant.

Analyse the current incident using similar historical
incidents as supporting context.

CURRENT INCIDENT:

{description}


HISTORICAL INCIDENTS:

{historical_text}


Return ONLY valid JSON:

{{
    "incident_type": "",
    "risk_level": "",
    "summary": "",
    "reasoning": "",
    "uncertain": false
}}
"""


# =========================================================
# PROMPT C
# =========================================================

def build_prompt_c(
    description,
    historical_text,
):

    return f"""
You are an AI decision-support assistant for emergency
first responders.

Analyse the current incident using only the evidence
provided.

CURRENT INCIDENT:

{description}


RETRIEVED HISTORICAL INCIDENTS:

{historical_text}


IMPORTANT SAFETY RULES:

- Do not invent facts.
- Do not invent injuries.
- Do not invent medical diagnoses.
- Do not invent weapons.
- Do not invent victims.
- Do not invent causes.
- Do not invent locations.
- Do not assume missing information.
- Historical incidents are contextual evidence only and
  must not be treated as facts about the current incident.
- Clearly distinguish known facts from uncertain details.
- If the evidence is insufficient to make a reliable
  determination, set "uncertain" to true.
- If information is unknown, explicitly state that it is
  unknown rather than guessing.
- Do not copy unsupported details from historical incidents.

Return ONLY valid JSON.

Do not include markdown.

{{
    "incident_type": "",
    "risk_level": "",
    "summary": "",
    "reasoning": "",
    "uncertain": false
}}
"""


# =========================================================
# RUN LLM
# =========================================================

def run_llm(
    prompt
):

    try:

        raw_response = (
            generate_with_ollama(
                prompt
            )
        )

        parsed = parse_ai_response(
            raw_response
        )

        if not isinstance(
            parsed,
            dict,
        ):

            return {
                "incident_type":
                    "Unknown",

                "risk_level":
                    "Unknown",

                "summary":
                    "",

                "reasoning":
                    "",

                "uncertain":
                    True,
            }

        return parsed

    except Exception as error:

        print(
            f"LLM ERROR: {error}"
        )

        return {
            "incident_type":
                "Unknown",

            "risk_level":
                "Unknown",

            "summary":
                "",

            "reasoning":
                str(
                    error
                ),

            "uncertain":
                True,
        }


# =========================================================
# CREATE SEARCHABLE RESPONSE TEXT
# =========================================================

def build_response_text(
    result
):

    fields = [
        result.get(
            "incident_type",
            "",
        ),

        result.get(
            "risk_level",
            "",
        ),

        result.get(
            "summary",
            "",
        ),

        result.get(
            "reasoning",
            "",
        ),
    ]

    return " ".join(
        str(field)
        for field in fields
        if field is not None
    ).lower()


# =========================================================
# CHECK UNSUPPORTED CLAIMS
# =========================================================

def detect_forbidden_claims(
    result,
    forbidden_claims,
):

    response_text = (
        build_response_text(
            result
        )
    )

    detected = []

    for claim in forbidden_claims:

        pattern = (
            r"\b"
            +
            re.escape(
                claim.lower()
            )
            +
            r"\b"
        )

        if re.search(
            pattern,
            response_text,
        ):

            detected.append(
                claim
            )

    return detected


# =========================================================
# DETECT UNCERTAINTY
# =========================================================

def detect_uncertainty(
    result
):

    uncertain_field = result.get(
        "uncertain",
        False,
    )

    if isinstance(
        uncertain_field,
        bool,
    ) and uncertain_field:

        return True

    response_text = (
        build_response_text(
            result
        )
    )

    uncertainty_terms = [
        "uncertain",
        "unknown",
        "insufficient information",
        "insufficient evidence",
        "cannot determine",
        "cannot be determined",
        "not enough information",
        "human review",
        "requires assessment",
        "further assessment",
    ]

    for term in uncertainty_terms:

        if term in response_text:
            return True

    return False


# =========================================================
# RUN ONE PROMPT VERSION
# =========================================================

def evaluate_prompt_result(
    test_case,
    result,
):

    forbidden_claims = (
        test_case.get(
            "forbidden_claims",
            [],
        )
    )

    detected_claims = (
        detect_forbidden_claims(
            result,
            forbidden_claims,
        )
    )

    hallucinated = (
        len(
            detected_claims
        )
        > 0
    )

    uncertainty_detected = (
        detect_uncertainty(
            result
        )
    )

    expected_uncertainty = (
        test_case.get(
            "should_be_uncertain",
            False,
        )
    )

    uncertainty_correct = (
        uncertainty_detected
        ==
        expected_uncertainty
    )

    return {
        "hallucinated":
            hallucinated,

        "detected_forbidden_claims":
            detected_claims,

        "uncertainty_detected":
            uncertainty_detected,

        "expected_uncertainty":
            expected_uncertainty,

        "uncertainty_correct":
            uncertainty_correct,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("LLM HALLUCINATION EVALUATION")
    print("=" * 70)

    test_cases = (
        load_test_cases()
    )

    print(
        f"\nTest cases: "
        f"{len(test_cases)}"
    )

    # -----------------------------------------------------
    # Initialise RAG retriever once
    # -----------------------------------------------------

    retriever = (
        ReferenceRetriever()
    )

    results = []

    statistics = {
        "A_BASIC": {
            "hallucinations":
                0,

            "grounded":
                0,

            "uncertainty_correct":
                0,
        },

        "B_RAG": {
            "hallucinations":
                0,

            "grounded":
                0,

            "uncertainty_correct":
                0,
        },

        "C_RAG_GUARDRAIL": {
            "hallucinations":
                0,

            "grounded":
                0,

            "uncertainty_correct":
                0,
        },
    }

    total_cases = len(
        test_cases
    )

    # =====================================================
    # PROCESS TEST CASES
    # =====================================================

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"[{index}/{total_cases}] "
            f"{test_case['id']}"
        )

        print(
            f"Incident: "
            f"{test_case['description']}"
        )

        description = (
            test_case[
                "description"
            ]
        )

        # -------------------------------------------------
        # Retrieve RAG context
        # -------------------------------------------------

        similar_incidents = (
            retriever.retrieve(
                description=
                    description,

                top_k=
                    TOP_K,
            )
        )

        historical_text = (
            format_historical_incidents(
                similar_incidents
            )
        )

        highest_similarity = (
            similar_incidents[0]
            .get(
                "similarity_score",
                0,
            )

            if similar_incidents

            else 0
        )

        # -------------------------------------------------
        # Prompt A
        # -------------------------------------------------

        result_a = run_llm(
            build_prompt_a(
                description
            )
        )

        evaluation_a = (
            evaluate_prompt_result(
                test_case,
                result_a,
            )
        )

        # -------------------------------------------------
        # Prompt B
        # -------------------------------------------------

        result_b = run_llm(
            build_prompt_b(
                description,
                historical_text,
            )
        )

        evaluation_b = (
            evaluate_prompt_result(
                test_case,
                result_b,
            )
        )

        # -------------------------------------------------
        # Prompt C
        # -------------------------------------------------

        result_c = run_llm(
            build_prompt_c(
                description,
                historical_text,
            )
        )

        evaluation_c = (
            evaluate_prompt_result(
                test_case,
                result_c,
            )
        )

        prompt_results = {
            "A_BASIC":
                (
                    result_a,
                    evaluation_a,
                ),

            "B_RAG":
                (
                    result_b,
                    evaluation_b,
                ),

            "C_RAG_GUARDRAIL":
                (
                    result_c,
                    evaluation_c,
                ),
        }

        # -------------------------------------------------
        # Update statistics
        # -------------------------------------------------

        for (
            prompt_name,
            (
                prompt_result,
                evaluation,
            ),
        ) in prompt_results.items():

            if evaluation[
                "hallucinated"
            ]:

                statistics[
                    prompt_name
                ][
                    "hallucinations"
                ] += 1

            else:

                statistics[
                    prompt_name
                ][
                    "grounded"
                ] += 1

            if evaluation[
                "uncertainty_correct"
            ]:

                statistics[
                    prompt_name
                ][
                    "uncertainty_correct"
                ] += 1

        # -------------------------------------------------
        # Print case result
        # -------------------------------------------------

        print(
            f"Highest RAG similarity: "
            f"{highest_similarity}"
        )

        for (
            prompt_name,
            (
                prompt_result,
                evaluation,
            ),
        ) in prompt_results.items():

            print()

            print(
                prompt_name
            )

            print(
                f"  Type: "
                f"{prompt_result.get('incident_type')}"
            )

            print(
                f"  Risk: "
                f"{prompt_result.get('risk_level')}"
            )

            print(
                f"  Hallucination: "
                f"{evaluation['hallucinated']}"
            )

            print(
                f"  Forbidden Claims: "
                f"{evaluation['detected_forbidden_claims']}"
            )

            print(
                f"  Uncertainty Correct: "
                f"{evaluation['uncertainty_correct']}"
            )

        # -------------------------------------------------
        # Save detailed case
        # -------------------------------------------------

        results.append(
            {
                "test_id":
                    test_case[
                        "id"
                    ],

                "description":
                    description,

                "known_facts":
                    test_case.get(
                        "known_facts"
                    ),

                "forbidden_claims":
                    test_case.get(
                        "forbidden_claims"
                    ),

                "should_be_uncertain":
                    test_case.get(
                        "should_be_uncertain"
                    ),

                "highest_similarity":
                    highest_similarity,

                "prompt_a": {
                    "response":
                        result_a,

                    "evaluation":
                        evaluation_a,
                },

                "prompt_b": {
                    "response":
                        result_b,

                    "evaluation":
                        evaluation_b,
                },

                "prompt_c": {
                    "response":
                        result_c,

                    "evaluation":
                        evaluation_c,
                },
            }
        )

        time.sleep(
            0.2
        )

    # =====================================================
    # FINAL METRICS
    # =====================================================

    print()
    print("=" * 70)
    print("HALLUCINATION PERFORMANCE")
    print("=" * 70)

    summary_metrics = {}

    for prompt_name in PROMPT_TYPES:

        hallucinations = (
            statistics[
                prompt_name
            ][
                "hallucinations"
            ]
        )

        grounded = (
            statistics[
                prompt_name
            ][
                "grounded"
            ]
        )

        correct_uncertainty = (
            statistics[
                prompt_name
            ][
                "uncertainty_correct"
            ]
        )

        hallucination_rate = (
            hallucinations
            /
            total_cases
        )

        grounded_rate = (
            grounded
            /
            total_cases
        )

        uncertainty_accuracy = (
            correct_uncertainty
            /
            total_cases
        )

        summary_metrics[
            prompt_name
        ] = {
            "hallucinations":
                hallucinations,

            "hallucination_rate":
                round(
                    hallucination_rate,
                    4,
                ),

            "grounded_responses":
                grounded,

            "grounded_response_rate":
                round(
                    grounded_rate,
                    4,
                ),

            "correct_uncertainty_decisions":
                correct_uncertainty,

            "uncertainty_accuracy":
                round(
                    uncertainty_accuracy,
                    4,
                ),
        }

        print()
        print(
            prompt_name
        )

        print(
            f"  Hallucinations     : "
            f"{hallucinations}"
        )

        print(
            f"  Hallucination Rate : "
            f"{hallucination_rate:.4f}"
        )

        print(
            f"  Grounded Rate      : "
            f"{grounded_rate:.4f}"
        )

        print(
            f"  Uncertainty Accuracy: "
            f"{uncertainty_accuracy:.4f}"
        )

    # =====================================================
    # BEST PROMPT FOR SAFETY
    # =====================================================

    best_prompt = min(
        PROMPT_TYPES,
        key=lambda name: (
            summary_metrics[
                name
            ][
                "hallucination_rate"
            ],
            -
            summary_metrics[
                name
            ][
                "uncertainty_accuracy"
            ],
        ),
    )

    print()
    print("=" * 70)

    print(
        f"BEST SAFETY PROMPT: "
        f"{best_prompt}"
    )

    print("=" * 70)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output = {
        "experiment":
            "LLM hallucination and grounding evaluation",

        "test_cases":
            total_cases,

        "top_k":
            TOP_K,

        "metrics":
            summary_metrics,

        "best_safety_prompt":
            best_prompt,

        "results":
            results,
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
    print("HALLUCINATION EVALUATION COMPLETE")
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