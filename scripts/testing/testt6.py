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

# Hybrid weights (tuned for SHL Recall@10)
W_FAISS = 0.45
W_INTENT = 0.25

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
# INTENT SCHEMA
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
# INTENT EXTRACTION
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
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vec = np.array(result.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# ============================================================
# INTENT ALIGNMENT SCORE (SIMPLIFIED)
# ============================================================

def intent_score(item, intent: HiringIntent) -> float:
    score = 0.0

    matched_types = set(intent.required_test_types) & set(item.get("test_types", []))
    score += min(len(matched_types), 2) * 2.0

    dur = item.get("duration_minutes")
    max_dur = intent.max_duration_minutes
    if max_dur and dur:
        if dur <= max_dur:
            score += 1.5
        elif dur <= max_dur + 15:
            score += 0.5

    return score

# ============================================================
# SHL CANONICAL BIAS (CRITICAL)
# ============================================================

def canonical_shl_bonus(item) -> float:
    name = item.get("assessment_name", "").lower()
    url = item.get("assessment_url", "").lower()

    keywords = [
        "verify",
        "verbal",
        "numerical",
        "inductive",
        "english comprehension",
        "opq",
        "personality questionnaire"
    ]

    if any(k in name or k in url for k in keywords):
        return 2.5

    return 0.0

# ============================================================
# ROLE SOLUTION ADJUSTMENT
# ============================================================

def role_solution_adjustment(item) -> float:
    name = item.get("assessment_name", "").lower()

    if "professional" in name:
        return 1.0

    if any(k in name for k in ["manager", "supervisor", "team lead"]):
        return -1.5

    return 0.0

# ============================================================
# FINAL HYBRID RANKING
# ============================================================

def rank_candidates(candidates, distances, intent: HiringIntent, final_k=10):
    ranked = []

    for i, item in enumerate(candidates):
        faiss_sim = float(distances[i])

        score = (
            W_FAISS * faiss_sim +
            W_INTENT * intent_score(item, intent) +
            canonical_shl_bonus(item) +
            role_solution_adjustment(item)
        )

        ranked.append((score, item))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:final_k]]

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
    D, I = index.search(query_vec.reshape(1, -1), TOP_K_RETRIEVAL)

    candidates = [catalog[i] for i in I[0]]
    distances = D[0]

    final = rank_candidates(candidates, distances, intent, FINAL_RESULTS)

    print("\nFINAL RECOMMENDATIONS (SHL-aligned):\n")

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
        """Content Writer required, expert in English and SEO."""
    )
