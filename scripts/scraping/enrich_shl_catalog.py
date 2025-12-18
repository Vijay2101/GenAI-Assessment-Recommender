import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# CONFIG

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MAX_RETRIES = 3
TIMEOUT = 10
RETRY_DELAY = 3
REQUEST_DELAY = 0.5

INPUT_FILE = os.path.join(
    "data", "raw", "shl_product_catalog_full.xlsx"
)

OUTPUT_FILE = os.path.join(
    "data", "processed", "shl_product_catalog_enriched.xlsx"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# HELPERS

def extract_field_by_label(soup, label):
    rows = soup.find_all(
        "div", class_="product-catalogue-training-calendar__row typ"
    )
    for row in rows:
        h4 = row.find("h4")
        if h4 and h4.get_text(strip=True) == label:
            p = row.find("p")
            return p.get_text(strip=True) if p else None
    return None


def scrape_detail_page_with_retry(url):
    if not url or pd.isna(url):
        return {
            "description": None,
            "job_levels": None,
            "languages": None,
            "assessment_length": None
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=HEADERS, timeout=TIMEOUT
            )
            if response.status_code != 200:
                raise Exception(
                    f"Status code {response.status_code}"
                )

            soup = BeautifulSoup(response.text, "html.parser")

            return {
                "description": extract_field_by_label(soup, "Description"),
                "job_levels": extract_field_by_label(soup, "Job levels"),
                "languages": extract_field_by_label(soup, "Languages"),
                "assessment_length": extract_field_by_label(
                    soup, "Assessment length"
                ),
            }

        except Exception as e:
            print(
                f"Retry {attempt}/{MAX_RETRIES} failed for {url} → {e}"
            )
            time.sleep(RETRY_DELAY)

    return {
        "description": None,
        "job_levels": None,
        "languages": None,
        "assessment_length": None
    }

# MAIN

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    df = pd.read_excel(INPUT_FILE)
    total = len(df)

    print(f"Enriching {total} assessments")

    enriched_rows = []

    for idx, row in df.iterrows():
        url = row.get("assessment_url")
        print(f"[{idx + 1}/{total}] Scraping → {url}")

        data = scrape_detail_page_with_retry(url)
        enriched_rows.append(data)

        time.sleep(REQUEST_DELAY)

    enriched_df = pd.DataFrame(enriched_rows)

    for col in enriched_df.columns:
        df[col] = enriched_df[col]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Enriched catalog saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
