from database import get_historical_incidents


incidents = get_historical_incidents(limit=5)

print("\nNumber returned:", len(incidents))

for index, incident in enumerate(incidents, start=1):
    print("\n" + "=" * 60)
    print(f"INCIDENT {index}")
    print("=" * 60)

    print(incident)

    print("\nAvailable fields:")
    print(list(incident.keys()))