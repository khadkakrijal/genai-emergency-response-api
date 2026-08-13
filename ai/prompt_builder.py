def build_emergency_prompt(
    incident_data: dict,
    similar_incidents: list
):
    """
    Build a structured prompt for emergency incident analysis.

    The prompt:
    - combines current incident information
    - includes retrieved historical incidents
    - provides clear risk-level rules
    - adds anti-hallucination instructions
    - requires structured JSON output
    """

    # -----------------------------------------------------
    # Format historical incidents
    # -----------------------------------------------------

    if similar_incidents:
        similar_text = "\n".join(
            [
                (
                    f"- Title: {i.get('title') or 'Unknown'} | "
                    f"Type: {i.get('incident_type') or 'Unknown'} | "
                    f"Description: {i.get('description') or 'Not provided'} | "
                    f"Similarity: {i.get('similarity_score', 0)}"
                )
                for i in similar_incidents
            ]
        )
    else:
        similar_text = (
            "No sufficiently relevant historical incidents "
            "were available."
        )

    # -----------------------------------------------------
    # Risk-level guidance
    # -----------------------------------------------------

    risk_rules = """
RISK LEVEL GUIDELINES

HIGH:
- Active weapon threat or violent assault.
- Serious or potentially serious injury.
- Vehicle collision where an injury has been reported.
- Active building or residential fire.
- Immediate threat to human life or public safety.
- Situation requiring urgent Police, Fire or Emergency Medical response.

MEDIUM:
- Medical symptoms requiring professional assessment where the person
  is conscious, stable and able to communicate.
- Non-life-threatening incident requiring emergency service assistance.
- Situation that may become more serious if it is not assessed.

LOW:
- Minor incident with no immediate threat to life or serious injury.
- Situation that does not require urgent emergency intervention.

UNCERTAIN:
- Available incident information is insufficient.
- Retrieved historical evidence is weak, irrelevant, or conflicting.
- A reliable risk level cannot be determined.
"""

    # -----------------------------------------------------
    # Safety / hallucination guardrails
    # -----------------------------------------------------

    safety_rules = """
IMPORTANT SAFETY RULES

- Use only information provided in the current incident and the supplied
  historical incidents.
- Do not invent injuries, weapons, victims, locations, responders,
  historical events, or other facts.
- Do not assume information that has not been provided.
- Historical incidents are supporting context only. They must not be
  treated as facts about the current incident.
- If the available information is insufficient, state that the result
  is uncertain.
- If historical evidence is weak or conflicting, lower the confidence.
- Do not give a confident recommendation when the supporting evidence
  is insufficient.
- When significant uncertainty exists, recommend human review.
"""

    return f"""
You are an AI decision-support assistant for emergency first responders.

Your role is to support human decision-making.
You must not replace the judgement of trained emergency personnel.

Analyse the current incident by combining all available information:

- description
- location
- incident time
- people involved
- weapon information
- injury information
- location type
- similar historical incidents

CURRENT INCIDENT

Description:
{incident_data.get("description") or "Not provided"}

Location:
{incident_data.get("location") or "Not provided"}

Incident Time:
{incident_data.get("incident_time") or "Not provided"}

People Involved:
{incident_data.get("people_involved") or "Not provided"}

Weapon Involved:
{incident_data.get("weapon_involved") or "Not provided"}

Injury Reported:
{incident_data.get("injury_reported") or "Not provided"}

Location Type:
{incident_data.get("location_type") or "Not provided"}


SIMILAR HISTORICAL INCIDENTS

{similar_text}


{risk_rules}


{safety_rules}


ANALYSIS INSTRUCTIONS

1. Determine the most appropriate incident type.

2. Determine the risk level using only:
   High, Medium, Low, or Uncertain.

3. If a weapon is actively being used to threaten another person,
   the risk level should normally be High.

4. If a vehicle collision includes a reported injury,
   the risk level should normally be High.

5. If an active residential or building fire is reported,
   the risk level should normally be High.

6. If a person has non-life-threatening medical symptoms,
   remains conscious, stable, and able to communicate,
   the risk level should normally be Medium.

7. Base the confidence score on the quality of the information
   and the relevance of the historical evidence.

8. Do not increase confidence merely because historical incidents exist.

9. If there is insufficient evidence to make a reliable determination,
   use:
   - incident_type: "Uncertain"
   - risk_level: "Uncertain"
   - low confidence_score
   - priority indicating human review

10. Explain your reasoning briefly using only the evidence provided.


Return ONLY valid JSON.

Do not include markdown.

Do not include ```json.

Do not include additional text before or after the JSON.


Return the JSON in exactly this structure:

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