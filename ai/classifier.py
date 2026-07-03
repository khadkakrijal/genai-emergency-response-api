from ai.retriever import retrieve_similar_incidents
from ai.summariser import generate_ai_analysis

class EmergencyClassifier:
    def classify(self, incident_data: dict):
        similar_incidents = retrieve_similar_incidents(
            incident_data["description"],
            top_k=5,
        )

        ai_result = generate_ai_analysis(
            incident_data=incident_data,
            similar_incidents=similar_incidents,
        )

        ai_result["similar_incidents"] = similar_incidents
        return ai_result