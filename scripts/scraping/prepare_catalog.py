import pandas as pd
import re
import ast
import os

# PATHS

INPUT_FILE = os.path.join(
    "data", "processed", "shl_product_catalog_enriched.xlsx"
)

OUTPUT_FILE = os.path.join(
    "data", "processed", "shl_product_catalog_prepared.xlsx"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# HELPERS

def parse_duration(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else None


TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}


def parse_test_types(val):
    if not isinstance(val, str):
        return []
    try:
        codes = ast.literal_eval(val)
        return [TEST_TYPE_MAP.get(c, c) for c in codes]
    except Exception:
        return []


def parse_list_field(text):
    if not isinstance(text, str):
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def yes_no_to_bool(val):
    if not isinstance(val, str):
        return None
    return val.lower() == "yes"


def clean_text(val):
    return val.strip() if isinstance(val, str) else None

# MAIN

def main():
    print("Loading enriched catalog...")
    df = pd.read_excel(INPUT_FILE)

    print("Cleaning text fields...")
    for col in ["description", "job_levels", "languages"]:
        df[col] = df[col].apply(clean_text)

    print("Parsing assessment duration...")
    df["duration_minutes"] = df["assessment_length"].apply(parse_duration)

    print("Mapping test type codes...")
    df["test_types"] = df["test_type_codes"].apply(parse_test_types)

    print("Parsing job levels & languages...")
    df["job_levels_list"] = df["job_levels"].apply(parse_list_field)
    df["languages_list"] = df["languages"].apply(parse_list_field)

    print("Normalizing boolean flags...")
    df["remote_testing_bool"] = df["remote_testing"].apply(yes_no_to_bool)
    df["adaptive_irt_bool"] = df["adaptive_irt"].apply(yes_no_to_bool)

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Prepared catalog saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
