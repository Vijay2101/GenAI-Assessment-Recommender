from fastapi import FastAPI
from app.schemas import RecommendResponse, RecommendedAssessment
from app.recommender import recommend
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import math

app = FastAPI(title="SHL GenAI Assessment Recommender",
              description="""
#  Live Application
### **🔗 [Open Frontend UI](https://assessment-recommender-frontend.vercel.app/)**
### **🔗 [Open Frontend Github repo](https://github.com/Vijay2101/Assessment-Recommender-Frontend)**
---

### ❗❗ FINAL PREDICTION SUBMISSION ❗❗
### **PLEASE USE THIS FILE AS THE FINAL PREDICTION SUBMISSION**
### **🔗 [CLICK HERE FOR FINAL SUBMISSION CSV FILE](https://github.com/Vijay2101/GenAI-Assessment-Recommender/blob/main/submission/vijay_kumar.csv)**
### **🔗 [CLICK HERE TO VIEW THE APPROACH DOCUMENT](https://github.com/Vijay2101/GenAI-Assessment-Recommender/blob/main/submission/SHL-GenAI-Assessment-Recommender.pdf)**

---
This API powers an AI-based (RAG) recommendation system that suggests relevant SHL assessments
based on a given job role or hiring requirement.

**For best experience, use the frontend UI.**  
The APIs below are exposed for testing and evaluation.
""")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(payload: dict):
    query = payload.get("query", "")

    _, results = recommend(query)

    formatted = []
    for r in results:
        dur = r.get("duration_minutes")

        # NaN-safe duration handling
        duration = (
            int(dur)
            if isinstance(dur, (int, float)) and not math.isnan(dur)
            else None
        )

        formatted.append(
            RecommendedAssessment(
                url=r["assessment_url"],
                name=r["assessment_name"],
                adaptive_support="Yes" if r.get("adaptive_irt") else "No",
                description=r.get("description", ""),
                duration=duration,
                remote_support="Yes" if r.get("remote_testing") else "No",
                test_type=r.get("test_types", [])
            )
        )

    return {
        "recommended_assessments": formatted
    }



@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SHL Assessment Recommendation API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f9fafb;
                color: #111827;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.08);
                max-width: 520px;
                text-align: center;
            }
            h1 {
                margin-bottom: 10px;
            }
            p {
                color: #6b7280;
                margin-bottom: 24px;
            }
            a {
                display: inline-block;
                margin: 8px;
                padding: 12px 20px;
                text-decoration: none;
                color: white;
                background: #2563eb;
                border-radius: 8px;
                font-weight: 500;
            }
            a.secondary {
                background: #4b5563;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SHL Assessment Recommendation API</h1>
            <p>
                This service powers an AI-based recommendation system for selecting
                relevant SHL assessments.
            </p>

            <a href="https://assessment-recommender-frontend.vercel.app/" target="_blank">
                Open Frontend UI
            </a>

            <a href="/docs" class="secondary">
                View API Docs
            </a>
        </div>
    </body>
    </html>
    """
