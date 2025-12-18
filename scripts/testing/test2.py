import os
import json
import time
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

load_dotenv()

FAISS_INDEX_PATH = "data/artifacts/faiss_index.bin"
METADATA_PATH = "data/artifacts/catalog_metadata.pkl"

EMBEDDING_MODEL = "gemini-embedding-001"
INTENT_MODEL = "gemini-2.5-flash"

TOP_K_RETRIEVAL = 40
FINAL_RESULTS = 10

# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading catalog metadata...")
with open(METADATA_PATH, "rb") as f:
    catalog = pickle.load(f)

print(f"Catalog size: {len(catalog)}")

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ============================================================
# Pydantic Schema (Structured Output)
# ============================================================

class HiringIntent(BaseModel):
    skills: List[str] = Field(description="Hard skills required")
    soft_skills: List[str] = Field(description="Soft skills required")
    max_duration_minutes: Optional[int] = Field(
        description="Maximum allowed duration in minutes"
    )
    job_level: Optional[str] = Field(
        description="Entry-level, Mid-Professional, Senior, Executive"
    )
    required_test_types: List[str] = Field(
        description="Assessment test types required"
    )

# ============================================================
# INTENT PROMPT
# ============================================================

INTENT_PROMPT = """
Extract hiring intent and constraints from the query.

Rules:
- Infer durations like "about an hour" = 60, "1-2 hours" = 120
- Allowed test types ONLY:
  Ability & Aptitude
  Knowledge & Skills
  Personality & Behavior
  Competencies
  Assessment Exercises
  Simulations
  Biodata & Situational Judgement
  Development & 360
- If something is not mentioned, return null or empty list
"""

# ============================================================
# INTENT EXTRACTION (STRUCTURED OUTPUT)
# ============================================================

def extract_intent(query: str) -> HiringIntent:
    response = client.models.generate_content(
        model=INTENT_MODEL,
        contents=f"{INTENT_PROMPT}\n\nQUERY:\n{query}",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": HiringIntent.model_json_schema(),
        },
    )
    return HiringIntent.model_validate_json(response.text)

# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(query: str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )
    vec = np.array(result.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# ============================================================
# FILTERING
# ============================================================

def filter_candidates(candidates, intent: HiringIntent):
    filtered = []

    for item in candidates:
        # Duration filter
        if intent.max_duration_minutes is not None:
            dur = item.get("duration_minutes")
            if dur is None or dur > intent.max_duration_minutes:
                continue

        # Job level filter
        if intent.job_level:
            if intent.job_level not in (item.get("job_levels") or []):
                continue

        filtered.append(item)

    return filtered

# ============================================================
# RANKING & BALANCING (ALL TEST TYPES)
# ============================================================

def rank_and_balance(
    candidates,
    required_test_types: List[str],
    final_k: int = 10
):
    if not required_test_types:
        return candidates[:final_k]

    buckets = {t: [] for t in required_test_types}

    for c in candidates:
        for t in required_test_types:
            if t in c.get("test_types", []):
                buckets[t].append(c)

    results = []
    while len(results) < final_k:
        progress = False
        for t in required_test_types:
            if buckets[t]:
                results.append(buckets[t].pop(0))
                progress = True
            if len(results) >= final_k:
                break
        if not progress:
            break

    return results

# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_test(query: str):
    print("\n" + "=" * 100)
    print("QUERY:")
    print(query)
    print("=" * 100)

    intent = extract_intent(query)
    print("\nEXTRACTED INTENT:")
    print(json.dumps(intent.model_dump(), indent=2))

    query_vec = embed_query(query)
    _, I = index.search(query_vec.reshape(1, -1), TOP_K_RETRIEVAL)

    candidates = [catalog[i] for i in I[0]]
    print(f"\nRetrieved candidates: {len(candidates)}")

    filtered = filter_candidates(candidates, intent)
    print(f"After filtering: {len(filtered)}")

    final = rank_and_balance(
        filtered,
        intent.required_test_types,
        FINAL_RESULTS
    )

    print("\nFINAL RECOMMENDATIONS (up to 10):\n")

    for i, item in enumerate(final, 1):
        print(f"{i}. {item['assessment_name']}")
        print(f"   URL: {item['assessment_url']}")
        print(f"   Duration: {item.get('duration_minutes')} mins")
        print(f"   Test Types: {item.get('test_types')}")
        print()

# ============================================================
# SAMPLE TEST CASES
# ============================================================

if __name__ == "__main__":
    # run_test(
    #     "I am hiring for Java developers who can collaborate effectively "
    #     "with my business teams. The assessment should be completed in 40 minutes."
    # )

    # # time.sleep(2)

    # # run_test(
    # #     "I want to hire new graduates for a sales role. "
    # #     "The budget is about an hour for each test."
    # # )

    # time.sleep(2)

    run_test(
        '''I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options'''
    )