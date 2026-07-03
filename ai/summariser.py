from ai.prompt_builder import build_emergency_prompt
from ai.ollama_service import generate_with_ollama
from ai.parser import parse_ai_response

def generate_ai_analysis(incident_data: dict, similar_incidents: list):
    prompt = build_emergency_prompt(incident_data, similar_incidents)
    raw_response = generate_with_ollama(prompt)
    return parse_ai_response(raw_response)