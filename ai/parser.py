import json


def parse_ai_response(raw_text: str):
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        json_text = raw_text[start:end]
        return json.loads(json_text)
    except Exception:
        return {
            "incident_type": "Unknown",
            "risk_level": "Unknown",
            "confidence_score": 0.0,
            "summary": raw_text,
            "recommended_response": "Review manually.",
            "key_risks": [],
            "reasoning": "AI response could not be parsed as JSON.",
        }