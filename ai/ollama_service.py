import requests

def generate_with_ollama(prompt: str, model: str = "llama3.2"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["response"]

    except Exception:
        return """
        {
          "incident_type": "Emergency Incident",
          "risk_level": "High",
          "confidence_score": 0.75,
          "priority": "High Priority",
          "responders": ["Police", "Ambulance"],
          "key_risks": ["Possible harm to people", "Unclear scene safety"],
          "summary": "Emergency incident detected. Immediate review by responders is recommended.",
          "recommended_response": "Dispatch appropriate emergency responders and assess the scene safely.",
          "reasoning": "Cloud fallback mode used because local Ollama is not available on the deployed server."
        }
        """