import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

# CONFIG

BASE_URL = "https://www.shl.com"

CATALOG_URL = "https://www.shl.com/products/product-catalog/?type=1"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shl_product_catalog_full.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# HELPERS

def yes_no_from_td(td):
    """Return Yes/No based on presence of -yes class"""
    span = td.find("span", class_="catalogue__circle")
    if not span:
        return "No"
    return "Yes" if "-yes" in span.get("class", []) else "No"


# SCRAPER

def scrape_individual_tests():
    all_data = []
    seen_pages = set()
    start = 1

    while True:
        url = f"{CATALOG_URL}&start={start}"
        print(f"Scraping: {url}")

        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print("Request failed, stopping.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.find_all("tr")

        if not rows:
            print("No rows found, stopping.")
            break

        current_page = []

        for row in rows:
            entity_id = row.get("data-entity-id")
            if not entity_id:
                continue

            tds = row.find_all("td")
            if len(tds) < 4:
                continue

            # ---- Title & URL ----
            title_td = row.find("td", class_="custom__table-heading__title")
            if not title_td:
                continue

            a_tag = title_td.find("a")
            if not a_tag:
                continue

            name = a_tag.get_text(strip=True)
            link = urljoin(BASE_URL, a_tag.get("href"))

            # ---- Remote Testing ----
            remote_support = yes_no_from_td(tds[1])

            # ---- Adaptive / IRT ----
            adaptive_support = yes_no_from_td(tds[2])

            # ---- Test Types ----
            test_type_td = row.find(
                "td",
                class_="custom__table-heading__general product-catalogue__keys"
            )

            test_types = []
            if test_type_td:
                for span in test_type_td.find_all(
                    "span", class_="product-catalogue__key"
                ):
                    test_types.append(span.get_text(strip=True))

            record = {
                "category": "Individual Test Solutions",
                "assessment_id": entity_id,
                "assessment_name": name,
                "assessment_url": link,
                "remote_testing": remote_support,
                "adaptive_irt": adaptive_support,
                "test_type_codes": test_types
            }

            current_page.append(record)

        if not current_page:
            print("No data on page, stopping.")
            break

        page_signature = tuple(r["assessment_id"] for r in current_page)
        if page_signature in seen_pages:
            print("Repeated page detected, stopping.")
            break

        seen_pages.add(page_signature)
        all_data.extend(current_page)
        start += 12

    return all_data


# MAIN

def main():
    records = scrape_individual_tests()
    df = pd.DataFrame(records)

    print(f"Total Individual Test Solutions scraped: {len(df)}")

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved catalog to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
