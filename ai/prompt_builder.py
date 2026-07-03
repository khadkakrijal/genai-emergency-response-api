def build_emergency_prompt(incident_data: dict, similar_incidents: list):
    similar_text = "\n".join([
        f"- {i.get('title')} | {i.get('incident_type')} | {i.get('description')}"
        for i in similar_incidents
    ])

    return f"""
You are an AI decision-support assistant for emergency first responders.

Analyse the current incident by fusing all available information:
- description
- location
- incident time
- people involved
- weapon information
- injury information
- location type
- similar historical incidents

Current incident:
Description: {incident_data.get("description")}
Location: {incident_data.get("location") or "Not provided"}
Incident Time: {incident_data.get("incident_time") or "Not provided"}
People Involved: {incident_data.get("people_involved") or "Not provided"}
Weapon Involved: {incident_data.get("weapon_involved") or "Not provided"}
Injury Reported: {incident_data.get("injury_reported") or "Not provided"}
Location Type: {incident_data.get("location_type") or "Not provided"}

Similar historical incidents:
{similar_text}

Return ONLY valid JSON in this format:
{{
  "incident_type": "",
  "risk_level": "",
  "confidence_score": 0.0,
  "priority": "",
  "responders": [],
  "key_risks": [],
  "summary": "",
  "recommended_response": "",
  "reasoning": ""
}}
"""