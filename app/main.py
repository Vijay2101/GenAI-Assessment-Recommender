from fastapi import FastAPI
from app.schemas import RecommendResponse, RecommendedAssessment
from app.recommender import recommend

app = FastAPI()


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(payload: dict):
    query = payload.get("query", "")

    _, results = recommend(query)

    formatted = []
    for r in results:
        formatted.append(
            RecommendedAssessment(
                url=r["assessment_url"],
                name=r["assessment_name"],
                adaptive_support="Yes" if r.get("adaptive_irt") else "No",
                description=r.get("description", ""),
                duration=int(r["duration_minutes"]) if r.get("duration_minutes") else 0,
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
