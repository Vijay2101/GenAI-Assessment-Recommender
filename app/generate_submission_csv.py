import os
import csv
import pandas as pd
from recommender import recommend

# PATH CONFIGURATION

EXCEL_PATH = os.path.join(
     "data", "test", "SHL_testSet.xlsx"
)

QUERY_COLUMN = "Query"

OUTPUT_DIR = os.path.join( "submission")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "submission_predictions.csv")

TOP_K = 10  

# PREP OUTPUT DIRECTORY
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD EXCEL QUERIES

df = pd.read_excel(EXCEL_PATH)

if QUERY_COLUMN not in df.columns:
    raise ValueError(
        f"Excel must contain a column named '{QUERY_COLUMN}'"
    )

queries = df[QUERY_COLUMN].dropna().tolist()
print(f"Loaded {len(queries)} queries from Excel")

# GENERATE SUBMISSION ROWS

rows = []

for idx, query in enumerate(queries, 1):
    print(f"Processing query {idx}/{len(queries)}")

    _, recommendations = recommend(query)

    for rec in recommendations[:TOP_K]:
        rows.append({
            "Query": query,
            "Assessment_url": rec["assessment_url"]
        })

# WRITE CSV (STRICT SHL FORMAT)

with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["Query", "Assessment_url"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n✅ Submission CSV successfully created")
print(f"📄 File location: {OUTPUT_CSV_PATH}")
