import os
import json
import time
import pickle
import numpy as np
import faiss
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configuration
load_dotenv()
CATALOG_PATH = "data/processed/catalog.json"
ARTIFACT_DIR = "data/artifacts"

FAISS_INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "catalog_metadata.pkl")

EMBEDDING_MODEL = "gemini-embedding-001"

BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 60.0  # seconds

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Gemini client

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Build embedding text

def build_embedding_text(item):
    parts = [
        item.get("assessment_name"),
        item.get("description"),
        " ".join(item.get("test_types", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join([p for p in parts if p])


# Embed batch (DOCUMENT embeddings)

def embed_batch(texts):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return [e.values for e in result.embeddings]


# Main pipeline

def main():
    print("Loading catalog JSON...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    texts = [build_embedding_text(item) for item in catalog]
    print(f"Total assessments: {len(texts)}")

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"Embedding batch {i} → {i + len(batch)}")

        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    embeddings = np.array(all_embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(catalog, f)

    print("Catalog embeddings & FAISS index built successfully")


if __name__ == "__main__":
    main()
