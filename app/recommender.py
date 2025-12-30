import os
import json
import pickle
import time
import re
import numpy as np
import faiss

from dotenv import load_dotenv
from typing import List, Optional
from functools import wraps
from pydantic import BaseModel, Field

from app.schemas import HiringIntent

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from rank_bm25 import BM25Okapi

# CONFIG

load_dotenv()

FAISS_INDEX_PATH = "data/artifacts/faiss_index.bin"
METADATA_PATH = "data/artifacts/catalog_metadata.pkl"

EMBEDDING_MODEL = "gemini-embedding-001"
INTENT_MODEL = "gemini-2.5-flash-lite"
RERANK_MODEL = "gemini-2.5-flash"

TOP_K_RETRIEVAL = 40
RERANK_K = 20
FINAL_RESULTS = 10

# GEMINI API KEY FALLBACK

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found in environment variables")

def gemini_fallback(func):
    """
    Decorator to retry Gemini API calls with fallback keys
    ONLY on 429 RESOURCE_EXHAUSTED
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None

        for idx, key in enumerate(GEMINI_KEYS):
            try:
                client = genai.Client(api_key=key)
                return func(client, *args, **kwargs)

            except ClientError as e:
                last_error = e
                if getattr(e, "code", None) == 429:
                    print(f"[WARN] Gemini key_{idx+1} rate-limited. Switching key...")
                    time.sleep(5)
                    continue
                else:
                    raise e

        raise RuntimeError("All Gemini API keys exhausted") from last_error

    return wrapper

# LOAD ARTIFACTS

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading catalog metadata...")
with open(METADATA_PATH, "rb") as f:
    catalog = pickle.load(f)

print(f"Catalog size: {len(catalog)}")

# BM25 SETUP

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

print("Building BM25 index...")

bm25_corpus = []
for item in catalog:
    text = " ".join([
        item.get("assessment_name", ""),
        item.get("description", ""),
        " ".join(item.get("test_types", [])),
        " ".join(item.get("skills", [])) if isinstance(item.get("skills"), list) else "",
        f"{item.get('duration_minutes', '')} minutes"
    ])
    bm25_corpus.append(tokenize(text))

bm25 = BM25Okapi(bm25_corpus)

def bm25_search(query: str, top_k: int):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [idx for idx in top_indices if scores[idx] > 0]

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
- Technical skills → Knowledge & Skills
- Communication / teamwork → Personality & Behavior
- Reasoning / aptitude → Ability & Aptitude
- Senior roles → Competencies

Allowed test types ONLY:
Ability & Aptitude
Knowledge & Skills
Personality & Behavior
Competencies
Assessment Exercises
Simulations
Biodata & Situational Judgement
Development & 360

Duration rules:
- "about an hour" = 60
- "1-2 hours" = 120

Return JSON only.
"""

# GEMINI CALLS

@gemini_fallback
def extract_intent_call(client, query: str):
    return client.models.generate_content(
        model=INTENT_MODEL,
        contents=f"{INTENT_PROMPT}\n\nQUERY:\n{query}",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": HiringIntent.model_json_schema(),
        },
    )

def extract_intent(query: str) -> HiringIntent:
    response = extract_intent_call(query)
    return HiringIntent.model_validate_json(response.text)

@gemini_fallback
def embed_query_call(client, query: str):
    return client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

def embed_query(query: str):
    result = embed_query_call(query)
    vec = np.array(result.embeddings[0].values, dtype="float32")
    faiss.normalize_L2(vec.reshape(1, -1))
    return vec

# HYBRID RETRIEVAL

def hybrid_retrieve(query: str):
    # FAISS
    query_vec = embed_query(query)
    _, I = index.search(query_vec.reshape(1, -1), TOP_K_RETRIEVAL)
    faiss_indices = list(I[0])

    # BM25
    bm25_indices = bm25_search(query, TOP_K_RETRIEVAL)

    # Merge
    merged = []
    seen = set()
    for idx in faiss_indices + bm25_indices:
        if idx not in seen:
            merged.append(catalog[idx])
            seen.add(idx)

    return merged

# GEMINI RERANKING

RERANK_PROMPT = """
You are ranking assessment products for a hiring manager.

Given:
- The original hiring query
- Extracted hiring intent
- A list of assessment options

Task:
Rank the assessments from MOST relevant to LEAST relevant.

STRICT RULES:
- Do NOT invent or remove items
- Do NOT explain reasoning
- Output ONLY a JSON array of indices (0-based)
- Indices must include ALL items exactly once

Focus on:
- Skill match
- Role relevance
- Duration constraints
"""

@gemini_fallback
def rerank_call(client, prompt: str):
    return client.models.generate_content(
        model=RERANK_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

def gemini_rerank(query, intent, candidates):
    items_text = []
    for i, c in enumerate(candidates):
        items_text.append(
            f"[{i}] Name: {c['assessment_name']}\n"
            f"Description: {c.get('description','')}\n"
            f"Duration: {c.get('duration_minutes')} mins\n"
            f"Test Types: {c.get('test_types')}\n"
        )

    prompt = f"""
{RERANK_PROMPT}

QUERY:
{query}

INTENT:
{json.dumps(intent.model_dump(), indent=2)}

ASSESSMENTS:
{chr(10).join(items_text)}
"""

    response = rerank_call(prompt)
    order = json.loads(response.text)
    return [candidates[i] for i in order]

# RUN RECOMMENDER

def recommend(query: str):
    """
    Core recommender function.
    Returns:
        intent (HiringIntent)
        final_results (List[dict])
    """

    intent = extract_intent(query)

    retrieved = hybrid_retrieve(query)
    if not retrieved:
        return intent, []

    rerank_candidates = retrieved[:RERANK_K]

    if len(rerank_candidates) == 1:
        final = rerank_candidates
    else:
        reranked = gemini_rerank(query, intent, rerank_candidates)
        final = reranked[:FINAL_RESULTS]

    return intent, final
