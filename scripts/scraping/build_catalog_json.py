import pandas as pd
import ast
import json
import os

INPUT_FILE = os.path.join(
    "data", "processed", "shl_product_catalog_prepared.xlsx"
)

OUTPUT_FILE = os.path.join(
    "data", "processed", "catalog.json"
)

def main():
    print("Loading prepared catalog...")
    df = pd.read_excel(INPUT_FILE)

    # Convert stringified lists back to lists
    for col in ["test_types", "job_levels_list", "languages_list"]:
        df[col] = df[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )

    # Rename for clean JSON schema
    df = df.rename(columns={
        "job_levels_list": "job_levels",
        "languages_list": "languages"
    })

    records = df.to_dict(orient="records")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Canonical JSON saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
