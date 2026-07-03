from pathlib import Path
import pandas as pd
from database import supabase

dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "911.csv"

df = pd.read_csv(dataset_path)

# Start small first
df = df.head(1000)

records = []

for _, row in df.iterrows():
    title = str(row["title"])
    category = title.split(":")[0].strip() if ":" in title else title

    records.append({
        "source": "Kaggle 911 Dataset",
        "title": title,
        "description": str(row["desc"]),
        "location": str(row["addr"]),
        "latitude": float(row["lat"]),
        "longitude": float(row["lng"]),
        "incident_type": category,
        "priority_level": None,
        "incident_time": str(row["timeStamp"]),
        "raw_data": {
            "zip": None if pd.isna(row["zip"]) else str(row["zip"]),
            "township": None if pd.isna(row["twp"]) else str(row["twp"]),
            "e": int(row["e"])
        }
    })

# Insert in batches
batch_size = 100

for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    response = supabase.table("historical_incidents").insert(batch).execute()
    print(f"Inserted batch {i // batch_size + 1}: {len(batch)} rows")

print("Import completed.")