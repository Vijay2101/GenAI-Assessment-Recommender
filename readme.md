SHL-GenAI-Assessment-Recommender
================================

A GenAI-powered Assessment Recommendation System that uses Retrieval-Augmented Generation (RAG) over SHL's product catalog to recommend the most relevant assessments for a given hiring query or job description.

* * * * *

Overview
--------

This project builds an end-to-end recommendation pipeline that:

-   Scrapes and processes SHL's public product catalog

-   Indexes assessments using vector embeddings

-   Understands user intent using a large language model

-   Retrieves, ranks, and balances assessment recommendations

-   Exposes a production-ready API for querying recommendations

-   Generates submission-ready evaluation files aligned with SHL's automated scoring pipeline


* * * * *

System Architecture
-------------------

**Pipeline Stages:**

1.  **Data Scraping**

    -   Scrapes SHL product catalog listing pages

    -   Extracts assessment metadata (name, URL, test types, flags)

    -   Visits individual product pages to extract descriptions, job levels, languages, and duration

2.  **Data Processing & Normalization**

    -   Cleans and standardizes text fields

    -   Maps test-type codes to canonical SHL categories

    -   Extracts numeric durations

    -   Converts data into a canonical JSON format

3.  **Embedding & Indexing**

    -   Uses `gemini-embedding-001` for document embeddings

    -   Stores normalized embeddings in a FAISS index

    -   Metadata stored separately for efficient retrieval

4.  **Query Understanding (LLM)**

    -   Uses `gemini-2.5-flash` with structured JSON outputs

    -   Extracts:

        -   Required test types

        -   Duration constraints

        -   Skills and soft skills

        -   Job seniority (if present)

5.  **Retrieval & Ranking**

    -   Vector similarity search (FAISS) to retrieve top-K candidates

    -   Soft scoring based on:

        -   Test type alignment

        -   Duration proximity

        -   Job-level relevance

    -   Balanced ranking across required test types


6.  **API Layer**

    -   FastAPI backend

    -   JSON-based recommendation responses

    -   Health check endpoint for deployment validation

* * * * *

Tech Stack
----------

-   **Backend:** FastAPI

-   **LLMs:** Google Gemini (genai SDK)

-   **Embeddings:** `gemini-embedding-001`

-   **Vector Store:** FAISS

-   **Scraping:** Requests, BeautifulSoup

-   **Data Processing:** Pandas, NumPy

-   **Deployment:** Vercel

* * * * *

API Endpoints
-------------

### Health Check

`GET /health`

**Response**

`{
  "status": "ok"
}`

* * * * *

### Recommendation Endpoint

`POST /recommend`

**Request Body**

```
{
  "query": "Content Writer required, expert in English and SEO."
}`

**Response**

`{
  "intent": {
    "required_test_types": ["Knowledge & Skills", "Personality & Behavior"],
    "max_duration_minutes": null
  },
  "recommendations": [
    {
      "assessment_name": "...",
      "assessment_url": "...",
      "duration_minutes": 25,
      "test_types": [...]
    }
  ]
}
```

* * * * *

Evaluation Methodology
----------------------

-   **Metric:** Mean Recall@10 (as specified by SHL)

-   **Approach:**

    -   Queries from the labeled training dataset were used to tune retrieval depth and ranking logic

    -   The system retrieves a candidate pool using vector similarity

    -   Final recommendations are ranked and balanced to maximize the chance of matching ground truth URLs in the top 10

-   **Automation:**

    -   An evaluation script generates a submission-ready CSV in the exact format required by SHL's automated scoring system

* * * * *

Submission CSV Format
---------------------

Generated automatically as:

```
Query,Assessment_url

Query 1,Recommendation URL 1

Query 1,Recommendation URL 2
```

The CSV is produced using queries read directly from the provided Excel test set.

* * * * *

Project Structure
-----------------

```
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

* * * * *

Deployment
----------

-   **Platform:** Vercel

-   **Root Directory:** `/app`

-   **Web App URL:**

    `https://gen-ai-assessment-recommender.vercel.app/docs`

Swagger UI is available at `/docs` for interactive testing.

* * * * *

How to Run Locally
------------------

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

* * * * *

Notes
-----

-   The system avoids pre-packaged job solutions and focuses on individual assessments, aligned with SHL's labeled datasets.

-   Rate limits are handled through batching and controlled request frequency.

-   The pipeline is fully reproducible and automated.

* * * * *

Author
------

**Vijay Kumar**