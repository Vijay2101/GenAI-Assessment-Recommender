import os
import json
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
    skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    job_level: Optional[str] = None
    required_test_types: List[str] = Field(default_factory=list)

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
# SCORING (SOFT CONSTRAINTS)
# ============================================================

def score_candidate(item, intent: HiringIntent) -> float:
    score = 0.0

    # 1️⃣ Test type relevance (highest weight)
    for t in intent.required_test_types:
        if t in (item.get("test_types") or []):
            score += 3.0

    # 2️⃣ Duration relevance (soft)
    dur = item.get("duration_minutes")
    max_dur = intent.max_duration_minutes

    if max_dur and dur:
        if dur <= max_dur:
            score += 2.0
        elif dur <= max_dur + 15:
            score += 1.0
    # missing duration → neutral

    # 3️⃣ Job level relevance (soft)
    if intent.job_level:
        if intent.job_level in (item.get("job_levels") or []):
            score += 1.0

    return score

# ============================================================
# RANKING (NO HARD FILTERING)
# ============================================================

def rank_candidates(candidates, intent: HiringIntent, final_k=10):
    scored = []

    for item in candidates:
        s = score_candidate(item, intent)
        scored.append((s, item))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer positive-score items
    results = [item for score, item in scored if score > 0]

    # Backfill if fewer than required
    if len(results) < final_k:
        results.extend(
            item for score, item in scored
            if item not in results
        )

    return results[:final_k]

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

    final = rank_candidates(candidates, intent, FINAL_RESULTS)

    print("\nFINAL RECOMMENDATIONS (up to 10):\n")

    for i, item in enumerate(final, 1):
        print(f"{i}. {item['assessment_name']}")
        print(f"   URL: {item['assessment_url']}")
        print(f"   Duration: {item.get('duration_minutes')} mins")
        print(f"   Test Types: {item.get('test_types')}")
        print()

# ============================================================
# SAMPLE TEST
# ============================================================

if __name__ == "__main__":
    run_test(
        # '''I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.'''
        '''I want to hire a Senior Data Analyst with 5 years of experience and expertise in SQL, Excel and Python. The assessment can be 1-2 hour long'''
    )
