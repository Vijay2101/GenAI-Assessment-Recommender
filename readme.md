SHL-GenAI-Assessment-Recommender
================================

A **GenAI-powered Assessment Recommendation System** that uses **Retrieval-Augmented Generation (RAG)** over SHL’s public product catalog to recommend the most relevant **SHL Individual Test Solutions** for a given hiring query or job description.

Overview
--------

This project implements an **end-to-end recommendation pipeline** that:

*   Collects and preprocesses SHL’s public product catalog
    
*   Builds reusable retrieval artifacts (FAISS index + metadata)
    
*   Understands hiring intent using a Large Language Model
    
*   Retrieves and reranks relevant assessments using a hybrid strategy
    
*   Exposes a production-ready API for querying recommendations
    
*   Generates submission-ready CSV files aligned with SHL’s evaluation format
    

System Architecture
-------------------

### Pipeline Stages

#### 1\. Data Collection

*   Programmatic extraction of SHL product catalog using **BeautifulSoup**
    
*   Retains only **Individual Test Solutions**
    
*   Extracts:
    
    *   assessment name and URL
        
    *   test type codes
        
    *   remote testing and adaptive (IRT) indicators
        

#### 2\. Data Processing & Normalization

*   Cleans and standardizes text fields
    
*   Converts assessment duration to numeric minutes
    
*   Maps test-type codes to canonical SHL categories
    
*   Produces a **canonical JSON dataset**
    

#### 3\. Embedding & Indexing (Offline)

*   Generates embeddings using **Gemini gemini-embedding-001**
    
*   Indexes embeddings using **FAISS (cosine similarity)**
    
*   Stores catalog metadata separately for efficient retrieval
    

#### 4\. Query Understanding (LLM)

*   Uses **Gemini gemini-2.5-flash-lite**
    
*   Extracts structured intent:
    
    *   required test types
        
    *   duration constraints
        
    *   skills and soft skills
        
    *   job seniority (if present)
        

#### 5\. Retrieval & Ranking

*   **Hybrid retrieval**:
    
    *   FAISS for semantic similarity
        
    *   BM25 for exact keyword matching
        
*   Candidates are reranked using **Gemini gemini-2.5-flash**
    
*   Ranking prioritizes:
    
    *   test-type alignment
        
    *   role relevance
        
    *   duration and job-level constraints
        

API Layer
---------

The system is exposed via a **FastAPI backend**.

### Health Check

**GET** /health

```bash
{ "status": "ok" }   
```

### Recommendation Endpoint

**POST** /recommend

**Request Body**

```json
   {    "query": "Content Writer required, expert in English and SEO."  }   
```

> **Note:**In Swagger UI, the request body may appear as{ "additionalProp1": {} } due to generic schema rendering.The actual request expects a JSON object with a single query field.

**Response**

```json
{
  "recommended_assessments": [
    {
      "name": "...",
      "url": "...",
      "description": "...",
      "duration": 25,
      "test_type": ["Knowledge & Skills"],
      "adaptive_support": "Yes",
      "remote_support": "Yes"
    }
  ]
}
```

Evaluation Methodology
----------------------

*   **Metric:** Mean Recall@10 (as specified by SHL)
    
*   **Approach:**
    
    *   Labeled training queries used to tune retrieval depth and ranking
        
    *   Candidate pool retrieved via hybrid retrieval
        
    *   Final top-10 recommendations optimized to maximize ground-truth overlap
        

### Submission CSV Format

Generated automatically as:

```bash  
Query, Assessment_url 
--------------------- 
Query 1,Recommendation URL 1  
Query 1,Recommendation URL 2   
```

The CSV is produced directly from the provided Excel test set and is **submission-ready**.

Project Structure
-----------------

```bash
├── app
│   ├── main.py              # FastAPI entrypoint
│   ├── recommender.py       # Core RAG logic
│   ├── schemas.py           # Pydantic schemas
│   └── generate_submission_csv.py
├── data
│   ├── raw                  # Scraped data
│   ├── processed            # Cleaned datasets
│   └── artifacts            # FAISS index & metadata
├── scripts
│   ├── scraping             # Web scraping pipeline
│   ├── pipeline             # End-to-end data pipeline
│   ├── indexing             # Embedding & FAISS indexing
│   └── testing              # Local testing scripts
├── submission               # Generated CSV output
├── requirements.txt
└── vercel.json
```

Deployment
----------

*   **Platform:** Vercel
    
*   **Root Directory:** /app
    
*   **Swagger UI:** [https://gen-ai-assessment-recommender.vercel.app/docs](https://gen-ai-assessment-recommender.vercel.app/docs)
    

Running Locally
---------------

Install dependencies:

```bash
   pip install -r requirements.txt   
```

Create environment file:

```bash
   cp .env.example .env   
```

Run the API:

```bash
   uvicorn app.main:app --reload   
```

Notes
-----

*   Gemini rate limits are handled via key rotation
    
*   The full pipeline is reproducible and automated
    

Author
------

**Vijay Kumar**