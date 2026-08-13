from ai.retriever import retrieve_similar_incidents
from ai.summariser import generate_ai_analysis


MIN_SIMILARITY_THRESHOLD = 0.10


class EmergencyClassifier:
    def classify(self, incident_data: dict):
        description = incident_data.get("description", "").strip()

        if not description:
            return self.human_review_response(
                reason="No incident description was provided."
            )

        similar_incidents = retrieve_similar_incidents(
            description,
            top_k=5,
        )

        if not similar_incidents:
            return self.human_review_response(
                reason="No historical incidents were available for comparison."
            )

        highest_similarity = max(
            float(item.get("similarity_score", 0))
            for item in similar_incidents
        )

        if highest_similarity < MIN_SIMILARITY_THRESHOLD:
            return self.human_review_response(
                reason=(
                    "Retrieved historical incidents did not meet "
                    "the minimum similarity threshold."
                ),
                similarity_score=highest_similarity,
                similar_incidents=similar_incidents,
            )

        try:
            ai_result = generate_ai_analysis(
                incident_data=incident_data,
                similar_incidents=similar_incidents,
            )

        except Exception as error:
            print(f"AI analysis error: {error}")

            return self.human_review_response(
                reason="AI analysis could not be completed.",
                similarity_score=highest_similarity,
                similar_incidents=similar_incidents,
            )

        if not isinstance(ai_result, dict):
            return self.human_review_response(
                reason="AI returned an invalid response.",
                similarity_score=highest_similarity,
                similar_incidents=similar_incidents,
            )

        required_fields = [
            "incident_type",
            "risk_level",
            "priority",
            "summary",
            "recommended_response",
            "reasoning",
        ]

        missing_fields = [
            field
            for field in required_fields
            if not ai_result.get(field)
        ]

        if missing_fields:
            return self.human_review_response(
                reason=(
                    "AI response was incomplete. Missing fields: "
                    + ", ".join(missing_fields)
                ),
                similarity_score=highest_similarity,
                similar_incidents=similar_incidents,
            )

        ai_result["similar_incidents"] = similar_incidents
        ai_result["highest_similarity_score"] = round(
            highest_similarity,
            4,
        )
        ai_result["needs_human_review"] = False
        ai_result["evidence_status"] = "Sufficient"

        return ai_result

    def human_review_response(
        self,
        reason: str,
        similarity_score: float = 0.0,
        similar_incidents: list | None = None,
    ):
        return {
            "incident_type": "Uncertain",
            "risk_level": "Uncertain",
            "priority": "Human Review Required",
            "confidence_score": 0.0,
            "responders": [],
            "key_risks": [],
            "summary": (
                "Insufficient reliable historical evidence "
                "is available to classify this incident."
            ),
            "recommended_response": (
                "A trained emergency operator should manually "
                "review this incident."
            ),
            "reasoning": reason,
            "needs_human_review": True,
            "evidence_status": "Insufficient",
            "highest_similarity_score": round(
                similarity_score,
                4,
            ),
            "similar_incidents": similar_incidents or [],
        }