import subprocess
import sys
import os

# PIPELINE STEPS
STEPS = [
    ("Scraping SHL catalog", "scripts/scraping/scrape_catalog.py"),
    ("Enriching assessment pages", "scripts/scraping/enrich_shl_catalog.py"),
    ("Preparing & normalizing data", "scripts/scraping/prepare_catalog.py"),
    ("Building canonical JSON", "scripts/scraping/build_catalog_json.py"),
    ("Generating embeddings & FAISS index", "scripts/indexing/build_embedding.py"),
]

# RUNNER

def run_step(name, script_path):
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"Step failed: {name}")
        sys.exit(1)

    print(f"Completed: {name}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(project_root)

    print("\nStarting FULL SHL DATA PIPELINE \n")

    for name, script in STEPS:
        run_step(name, script)

    print("\n PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
