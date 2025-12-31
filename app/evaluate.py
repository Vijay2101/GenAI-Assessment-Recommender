import pandas as pd
from collections import defaultdict
from typing import List
from recommender import recommend

# CONFIG

TRAIN_CSV_PATH = "data/test/SHL_trainSet.csv"   # update path if needed
K = 10

# LOAD TRAIN DATA

def load_ground_truth(csv_path: str):
    """
    Loads train CSV and groups ground-truth URLs by query.

    Expected CSV format:
    Query,Assessment_url
    """
    df = pd.read_csv(csv_path)

    if "Query" not in df.columns or "Assessment_url" not in df.columns:
        raise ValueError("Train CSV must contain 'Query' and 'Assessment_url' columns")

    ground_truth = defaultdict(list)
    for _, row in df.iterrows():
        query = row["Query"]
        url = row["Assessment_url"]
        ground_truth[query].append(url)

    return ground_truth


# METRIC: RECALL@K 

def extract_assessment_id(url: str) -> str:
    """
    Extracts stable assessment identifier from SHL URLs.
    Example:
    https://www.shl.com/.../view/java-8-new/  -> java-8-new
    """
    return url.split("view/")[-1].rstrip("/").lower()


def recall_at_k(predicted_urls: List[str], true_urls: List[str], k: int) -> float:
    """
    Recall@K = (# relevant assessments in top K) / (total relevant assessments)
    """
    if not true_urls:
        return 0.0

    predicted_ids = {
        extract_assessment_id(u)
        for u in predicted_urls[:k]
    }

    true_ids = {
        extract_assessment_id(u)
        for u in true_urls
    }

    return len(predicted_ids & true_ids) / len(true_ids)


# EVALUATION LOOP

def evaluate_mean_recall(train_csv: str, k: int = 10):
    ground_truth = load_ground_truth(train_csv)

    recalls = []
    total_queries = len(ground_truth)

    print(f"Evaluating Mean Recall@{k} on {total_queries} training queries...\n")

    for idx, (query, true_urls) in enumerate(ground_truth.items(), start=1):
        try:
            _, recommendations = recommend(query)
        except Exception as e:
            print(f"[WARN] Recommendation failed for query {idx}: {e}")
            recalls.append(0.0)
            continue

        predicted_urls = [
            r["assessment_url"] for r in recommendations
            if "assessment_url" in r
        ]

        r = recall_at_k(predicted_urls, true_urls, k)
        print("-" * 60)
        print(f"Predicted URLs: {predicted_urls}")
        print(f"True URLs: {true_urls}")
        print("=" * 60)
        print(f"Query {idx}/{total_queries}: Recall@{k} = {r:.4f}")
        print("=" * 60)
        recalls.append(r)

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

    print("=" * 60)
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print("=" * 60)

    return mean_recall


if __name__ == "__main__":
    evaluate_mean_recall(TRAIN_CSV_PATH, K)
