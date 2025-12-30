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
# INTENT SCHEMA
# ============================================================

class HiringIntent(BaseModel):
    skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    max_duration_minutes: Optional[int] = None
    job_level: Optional[str] = None
    required_test_types: List[str] = Field(default_factory=list)

# ============================================================
# INTENT PROMPT (CRITICAL FIX)
# ============================================================

INTENT_PROMPT = """
Extract hiring intent and constraints from the query.

IMPORTANT:
The query may contain MULTIPLE independent requirements.
You MUST identify ALL relevant assessment dimensions.

Rules:
- If technical skills (programming, tools, analytics) are mentioned,
  include "Knowledge & Skills".
- If collaboration, communication, teamwork, stakeholders, leadership,
  or culture is mentioned,
  include "Personality & Behavior".
- If reasoning, aptitude, or cognitive ability is mentioned,
  include "Ability & Aptitude".
- If senior / leadership roles are mentioned,
  include "Competencies".

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
- "40 minutes" = 40
- "about an hour" = 60
- "1-2 hours" = 120

Output MUST include ALL applicable test types.
Do NOT prioritize one over another.
If something is not mentioned, return null or empty list.
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

    # Test type relevance (strong)
    for t in intent.required_test_types:
        if t in (item.get("test_types") or []):
            score += 3.0

    # Duration relevance (soft)
    dur = item.get("duration_minutes")
    max_dur = intent.max_duration_minutes
    if max_dur and dur:
        if dur <= max_dur:
            score += 2.0
        elif dur <= max_dur + 15:
            score += 1.0

    # Job level relevance (soft)
    if intent.job_level:
        if intent.job_level in (item.get("job_levels") or []):
            score += 1.0

    return score

# ============================================================
# BALANCED RANKING (KEY FIX)
# ============================================================

def rank_candidates(candidates, intent: HiringIntent, final_k=10):
    # Fallback if no test types detected
    if not intent.required_test_types:
        scored = [(score_candidate(c, intent), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:final_k]]

    # Bucket by test type
    buckets = {t: [] for t in intent.required_test_types}

    for c in candidates:
        for t in intent.required_test_types:
            if t in (c.get("test_types") or []):
                buckets[t].append(c)

    # Sort each bucket by score
    for t in buckets:
        buckets[t].sort(
            key=lambda c: score_candidate(c, intent),
            reverse=True
        )

    results = []
    per_type_quota = max(1, final_k // len(intent.required_test_types))

    # First pass: enforce balance
    for t in intent.required_test_types:
        results.extend(buckets[t][:per_type_quota])

    # Second pass: fill remaining slots
    if len(results) < final_k:
        remaining = []
        for t in buckets:
            remaining.extend(buckets[t][per_type_quota:])

        # Deduplicate
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

    print("\nFINAL RECOMMENDATIONS (balanced, up to 10):\n")

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
        '''IFind me 1 hour long assesment for the below job at SHL
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

SHL is an equal opportunity employer.''')