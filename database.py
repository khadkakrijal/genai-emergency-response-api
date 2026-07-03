import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase URL or key in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_incident(incident_data: dict):
    response = supabase.table("incidents").insert(incident_data).execute()
    return response.data


def get_recent_incidents(limit: int = 5):
    response = (
        supabase.table("incidents")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data

def get_historical_incidents(limit: int = 1000):
    response = (
        supabase.table("historical_incidents")
        .select("*")
        .limit(limit)
        .execute()
    )
    return response.data