import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

historical_incidents = [
    {
        "source": "sample",
        "title": "Armed threat in residential area",
        "description": "A male suspect threatened a person with a knife near a residential street.",
        "location": "Nightcliff",
        "incident_type": "Armed Assault",
        "priority_level": "High",
    },
    {
        "source": "sample",
        "title": "Domestic dispute with possible weapon",
        "description": "Caller reported shouting and a possible knife involved during a domestic disturbance.",
        "location": "Coconut Grove",
        "incident_type": "Domestic Violence",
        "priority_level": "High",
    },
    {
        "source": "sample",
        "title": "Vehicle crash with injury",
        "description": "Two vehicles collided at an intersection and one person appeared injured.",
        "location": "Darwin City",
        "incident_type": "Traffic Accident",
        "priority_level": "Medium",
    },
    {
        "source": "sample",
        "title": "House fire reported",
        "description": "Smoke and flames were reported coming from a residential property.",
        "location": "Palmerston",
        "incident_type": "Fire Emergency",
        "priority_level": "High",
    },
    {
        "source": "sample",
        "title": "Medical emergency",
        "description": "A person collapsed and was unresponsive at a public location.",
        "location": "Casuarina",
        "incident_type": "Medical Emergency",
        "priority_level": "High",
    },
]

response = supabase.table("historical_incidents").insert(historical_incidents).execute()

print("Inserted historical incidents successfully.")
print(response)