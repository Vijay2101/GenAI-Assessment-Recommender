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

from functools import wraps
from google.genai.errors import ClientError
import time

GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]

GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

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



FAISS_INDEX_PATH = "data/artifacts/faiss_index.bin"
METADATA_PATH = "data/artifacts/catalog_metadata.pkl"

EMBEDDING_MODEL = "gemini-embedding-001"
# INTENT_MODEL = "gemini-2.5-flash"
INTENT_MODEL = "gemini-2.5-flash-lite"
RERANK_MODEL = "gemini-2.5-flash"

TOP_K_RETRIEVAL = 40
RERANK_K = 20
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

from rank_bm25 import BM25Okapi
import re

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
        " ".join(item.get("test_types", []))
    ])
    bm25_corpus.append(tokenize(text))

bm25 = BM25Okapi(bm25_corpus)

def bm25_search(query: str, top_k: int):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

def hybrid_retrieve(query: str):
    # ---- FAISS ----
    query_vec = embed_query(query)
    _, I = index.search(query_vec.reshape(1, -1), TOP_K_RETRIEVAL)
    faiss_results = list(I[0])

    # ---- BM25 ----
    bm25_results = bm25_search(query, TOP_K_RETRIEVAL)
    bm25_indices = [idx for idx, _ in bm25_results]

    # ---- MERGE (preserve order bias) ----
    merged = []
    seen = set()

    for idx in faiss_results + bm25_indices:
        if idx not in seen:
            merged.append(catalog[idx])
            seen.add(idx)

    return merged


# ============================================================
# GEMINI CLIENT
# ============================================================

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

# ============================================================
# INTENT EXTRACTION
# ============================================================

# def extract_intent(query: str) -> HiringIntent:
#     response = client.models.generate_content(
#         model=INTENT_MODEL,
#         contents=f"{INTENT_PROMPT}\n\nQUERY:\n{query}",
#         config={
#             "response_mime_type": "application/json",
#             "response_json_schema": HiringIntent.model_json_schema(),
#         },
#     )
#     return HiringIntent.model_validate_json(response.text)

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



# ============================================================
# QUERY EMBEDDING
# ============================================================

# def embed_query(query: str):
#     result = client.models.embed_content(
#         model=EMBEDDING_MODEL,
#         contents=query,
#         config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
#     )
#     vec = np.array(result.embeddings[0].values, dtype="float32")
#     faiss.normalize_L2(vec.reshape(1, -1))
#     return vec

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
# ============================================================
# GEMINI RERANKER (CORE FIX)
# ============================================================

RERANK_PROMPT = """
You are ranking assessment products for a hiring manager.

Given:
- The original hiring query
- Extracted hiring intent
- A list of assessment options

Task:
Rank the assessments from MOST relevant to LEAST relevant.
The ranking should be overall relevance to the hiring query.
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
    print(prompt)

    order = json.loads(response.text)
    
    print(f"\nRerank order: {order}")
    return [candidates[i] for i in order]

# ============================================================
# MAIN RUNNER
# ============================================================

def run_test(query: str):
    print("\n" + "=" * 100)
    print("QUERY:")
    print(query)
    print("=" * 100)

    intent = extract_intent(query)
    print("\nEXTRACTED INTENT:")
    print(json.dumps(intent.model_dump(), indent=2))

    # query_vec = embed_query(query)
    # _, I = index.search(query_vec.reshape(1, -1), TOP_K_RETRIEVAL)

    # retrieved = [catalog[i] for i in I[0]]
    retrieved = hybrid_retrieve(query)
    print(f"\nRetrieved (FAISS + BM25): {len(retrieved)}")

    # print(f"\nRetrieved from FAISS: {len(retrieved)}")

    rerank_candidates = retrieved[:RERANK_K]
    reranked = gemini_rerank(query, intent, rerank_candidates)

    final = reranked[:FINAL_RESULTS]

    print("\nFINAL RECOMMENDATIONS (Gemini-ranked):\n")
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
        '''Find me 1 hour long assesment for the below job at SHL
Job Description

 Join a community that is shaping the future of work! SHL, People Science. People Answers. 

Are you a seasoned QA Engineer with a flair for innovation? Are you ready to shape the future of talent assessment and empower organizations to unlock their full potential? If so, we want you to be a part of the SHL Team! As a QA Engineer, you will be involved in creating and implementing software solutions that contribute to the development of our ground-breaking products.

An excellent benefit package is offered in a culture where career development, with ongoing manager guidance, collaboration, flexibility, diversity, and inclusivity are all intrinsic to our culture.  There is a huge investment in SHL currently so there’s no better time to become a part of something transformational.

What You Will Be Doing

 Getting involved in engineering quality assurance and providing inputs when required. 
 Create and develop test plans for various forms of testing. 
 Conducts and/or participates in formal and informal test case reviews. 
 Develop and initiate functional tests and regression tests. 
 Rolling out improvements for testing and quality processes. 

Essential

 What we are looking for from you: 

 Development experience – Java or JavaScript, CSS, HTML (Automation) 
 Selenium WebDriver and page object design pattern (Automation) 
 SQL server knowledge 
 Test case management experience. 
 Manual Testing 

Desirable

 Knowledge the basic concepts of testing 
 Strong solution-finding experience 
 Strong verbal and written communicator. 

Get In Touch

Find out how this one-off opportunity can help you achieve your career goals by making an application to our knowledgeable and friendly Talent Acquisition team. Choose a new path with SHL.

 #CareersAtSHL #SHLHiringTalent 

#TechnologyJobs #QualityAssuranceJobs

#CareerOpportunities #JobOpportunities 

About Us

We unlock the possibilities of businesses through the power of people, science and technology.
We started this industry of people insight more than 40 years ago and continue to lead the market with powerhouse product launches, ground-breaking science and business transformation.
When you inspire and transform people’s lives, you will experience the greatest business outcomes possible. SHL’s products insights, experiences, and services can help achieve growth at scale.

What SHL Can Offer You

Diversity, equity, inclusion and accessibility are key threads in the fabric of SHL’s business and culture (find out more about DEI and accessibility at SHL )
Employee benefits package that takes care of you and your family.
Support, coaching, and on-the-job development to achieve career success
A fun and flexible workplace where you’ll be inspired to do your best work (find out more LifeAtSHL )
The ability to transform workplaces around the world for others.

SHL is an equal opportunity employer. We support and encourage applications from a diverse range of candidates. We can, and do make adjustments to make sure our recruitment process is as inclusive as possible.

SHL is an equal opportunity employer.'''
    )