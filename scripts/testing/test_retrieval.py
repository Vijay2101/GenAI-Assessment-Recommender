import os
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -------------------------
# Config
# -------------------------
load_dotenv()

FAISS_INDEX_PATH = "data/artifacts/faiss_index.bin"
METADATA_PATH = "data/artifacts/catalog_metadata.pkl"

EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K_RETRIEVAL = 40        # retrieve more, filter later
FINAL_RESULTS = 6           # show 5–10 typically

# -------------------------
# Load artifacts
# -------------------------

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading catalog metadata...")
with open(METADATA_PATH, "rb") as f:
    catalog = pickle.load(f)

print(f"Catalog size: {len(catalog)}")

# -------------------------
# Gemini client
# -------------------------

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------
# Query embedding
# -------------------------

def embed_query(query: str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    return np.array(result.embeddings[0].values, dtype="float32")

# -------------------------
# Filtering logic
# -------------------------

def filter_candidates(candidates, max_duration=None):
    filtered = []

    for item in candidates:
        if max_duration is not None:
            if item.get("duration_minutes") is None:
                continue
            if item["duration_minutes"] > max_duration:
                continue

        filtered.append(item)

    return filtered

# -------------------------
# Ranking & balancing
# -------------------------

def rank_and_balance(candidates):
    """
    Simple balanced ranking:
    - Prefer mix of Knowledge & Skills + Personality/Competencies
    """
    tech = []
    soft = []

    for c in candidates:
        types_ = c.get("test_types", [])
        if "Knowledge & Skills" in types_:
            tech.append(c)
        elif any(t in types_ for t in ["Personality & Behavior", "Competencies"]):
            soft.append(c)

    results = []

    # Alternate tech & soft if possible
    while len(results) < FINAL_RESULTS and (tech or soft):
        if tech:
            results.append(tech.pop(0))
        if len(results) >= FINAL_RESULTS:
            break
        if soft:
            results.append(soft.pop(0))

    return results

# -------------------------
# Main test
# -------------------------

def run_test(query, max_duration=None):
    print("\nQUERY:", query)
    print("-" * 80)

    query_embedding = embed_query(query)
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    D, I = index.search(query_embedding.reshape(1, -1), TOP_K_RETRIEVAL)

    candidates = [catalog[i] for i in I[0]]

    print(f"Retrieved {len(candidates)} candidates")

    filtered = filter_candidates(candidates, max_duration=max_duration)
    print(f"After filtering: {len(filtered)}")

    final = rank_and_balance(filtered)

    print("\nFINAL RECOMMENDATIONS:\n")

    for idx, item in enumerate(final, 1):
        print(f"{idx}. {item['assessment_name']}")
        print(f"   URL: {item['assessment_url']}")
        print(f"   Duration: {item.get('duration_minutes')} mins")
        print(f"   Types: {item.get('test_types')}")
        print()

# -------------------------
# Run example tests
# -------------------------

if __name__ == "__main__":
    # run_test(
    #     query="I am hiring for Java developers who can collaborate effectively with my business teams.",
    #     max_duration=40
    # )

    run_test(
        query="I am looking for a COO for my company in China and I want to see if they are culturally a right fit for our company. Suggest me an assessment that they can complete in about an hour",
        max_duration=60
    )
