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
import time
from typing import Callable

# CONFIG

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data/artifacts/faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "data/artifacts/catalog_metadata.pkl")

EMBEDDING_MODEL = "gemini-embedding-001"
INTENT_MODEL = "gemini-2.5-flash"

TOP_K_RETRIEVAL = 40
FINAL_RESULTS = 10


def with_retry(fn: Callable, retries: int = 3, wait_seconds: int = 30):
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < retries:
                print(f"[Retry] Attempt {attempt} failed. Retrying in {wait_seconds}s...")
                time.sleep(wait_seconds)
            else:
                print("[Retry] All attempts failed.")
    raise last_exception


# LOAD ARTIFACTS (ONCE)

index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    catalog = pickle.load(f)

# GEMINI CLIENT

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# INTENT SCHEMA

class HiringIntent(BaseModel):
    skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    job_level: Optional[str] = None
    required_test_types: List[str] = Field(default_factory=list)

# INTENT PROMPT

INTENT_PROMPT = """
Extract hiring intent and constraints from the query.

Rules:
- Include ALL applicable assessment dimensions
- Allowed test types ONLY:
  Ability & Aptitude
  Knowledge & Skills
  Personality & Behavior
  Competencies
  Assessment Exercises
  Simulations
  Biodata & Situational Judgement
  Development & 360

Duration rules:
- "40 minutes" = 40
- "about an hour" = 60
- "1-2 hours" = 120
"""

# INTENT EXTRACTION

def extract_intent(query: str) -> HiringIntent:
    def _call():
        response = client.models.generate_content(
            model=INTENT_MODEL,
            contents=f"{INTENT_PROMPT}\n\nQUERY:\n{query}",
            config={
                "response_mime_type": "application/json",
                "response_json_schema": HiringIntent.model_json_schema(),
            },
        )
        return HiringIntent.model_validate_json(response.text)

    return with_retry(_call, retries=3, wait_seconds=30)


# QUERY EMBEDDING

def embed_query(query: str):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vec = np.array(result.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# SCORING

def score_candidate(item, intent: HiringIntent) -> float:
    score = 0.0

    for t in intent.required_test_types:
        if t in (item.get("test_types") or []):
            score += 3.0

    dur = item.get("duration_minutes")
    max_dur = intent.max_duration_minutes
    if max_dur and dur:
        if dur <= max_dur:
            score += 2.0
        elif dur <= max_dur + 15:
            score += 1.0

    if intent.job_level:
        if intent.job_level in (item.get("job_levels") or []):
            score += 1.0

    return score

# RANKING

def rank_candidates(candidates, intent: HiringIntent):
    buckets = {t: [] for t in intent.required_test_types}

    for c in candidates:
        for t in intent.required_test_types:
            if t in (c.get("test_types") or []):
                buckets[t].append(c)

    for t in buckets:
        buckets[t].sort(
            key=lambda c: score_candidate(c, intent),
            reverse=True
        )

    results = []
    per_type = max(1, FINAL_RESULTS // max(1, len(intent.required_test_types)))

    for t in buckets:
        results.extend(buckets[t][:per_type])

    if len(results) < FINAL_RESULTS:
        remaining = []
        for t in buckets:
            remaining.extend(buckets[t][per_type:])

        seen = set()
        deduped = []
        for r in remaining:
            if r["assessment_url"] not in seen:
                deduped.append(r)
                seen.add(r["assessment_url"])

        deduped.sort(
            key=lambda c: score_candidate(c, intent),
            reverse=True
        )
        results.extend(deduped)

    return results[:FINAL_RESULTS]


# MAIN RECOMMENDER

def recommend(query: str):
    intent = extract_intent(query)
    q_vec = embed_query(query)

    _, I = index.search(q_vec.reshape(1, -1), TOP_K_RETRIEVAL)
    candidates = [catalog[i] for i in I[0]]

    final = rank_candidates(candidates, intent)

    return intent.model_dump(), final
