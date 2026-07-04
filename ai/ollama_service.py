import os
import requests
from groq import Groq


def generate_with_ollama(prompt: str, model: str = "llama3.2"):
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"

    if use_groq:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an emergency response AI assistant. Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        return response.choices[0].message.content

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["response"]