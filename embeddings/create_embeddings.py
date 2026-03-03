# ============================================
# SHL Assessment Embedding + Vector DB Builder
# ============================================

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading datasets...")

# ─────────────────────────────────────────────
# Resolve paths relative to project root
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# 1) LOAD + MERGE CSV FILES
# --------------------------------------------------

df1 = pd.read_csv(os.path.join(BASE_DIR, "data", "data_product_links.csv"))
df2 = pd.read_csv(os.path.join(BASE_DIR, "data", "shl_assessments_final.csv"))

# Use df2 (rich data: name, test_type, description) as primary.
# Only append df1 rows whose URLs are NOT already in df2.
extra_urls = df1[~df1["url"].isin(df2["url"])].copy()
df = pd.concat([df2, extra_urls], ignore_index=True)
df.drop_duplicates(subset=["url"], keep="first", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Total assessments after merge: {len(df)} (df2={len(df2)}, extra from df1={len(extra_urls)})")


# --------------------------------------------------
# 2) BASIC DATA CLEANING
# --------------------------------------------------

df = df.fillna("Not Available")

required_columns = [
    "name", "url", "description",
    "duration", "test_type", "adaptive_support", "remote_support"
]

for col in required_columns:
    if col not in df.columns:
        df[col] = "Not Available"

print("Columns ready ✔")


# --------------------------------------------------
# 3) CREATE TEXT FOR EMBEDDING
# --------------------------------------------------

def create_embedding_text(row):
    return (
        f"Assessment Name: {row['name']}\n"
        f"Description: {row['description']}\n"
        f"Test Type: {row['test_type']}\n"
        f"Duration: {row['duration']}\n"
        f"Adaptive Support: {row['adaptive_support']}\n"
        f"Remote Support: {row['remote_support']}"
    )

df["embedding_text"] = df.apply(create_embedding_text, axis=1)
texts = df["embedding_text"].tolist()

print("Text prepared for embeddings ✔")


# --------------------------------------------------
# 4) LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Embeddings created ✔")


# --------------------------------------------------
# 5) CREATE CHROMA VECTOR DATABASE (PersistentClient)
# --------------------------------------------------

db_path = os.path.join(BASE_DIR, "vector_db")
print(f"Creating Chroma vector database at '{db_path}' ...")

client = chromadb.PersistentClient(path=db_path)

# Delete existing collection so we can rebuild cleanly
try:
    client.delete_collection("shl_assessments")
except Exception:
    pass

collection = client.create_collection("shl_assessments")


# --------------------------------------------------
# 6) STORE EMBEDDINGS IN BATCHES
# --------------------------------------------------

print("Storing embeddings in vector database...")

BATCH_SIZE = 100
total = len(df)

for start in range(0, total, BATCH_SIZE):
    end   = min(start + BATCH_SIZE, total)
    batch = df.iloc[start:end]

    collection.add(
        documents  = texts[start:end],
        embeddings = [e.tolist() for e in embeddings[start:end]],
        metadatas  = [
            {
                "name":             str(row["name"]),
                "url":              str(row["url"]),
                "description":      str(row["description"]),
                "duration":         str(row["duration"]),
                "test_type":        str(row["test_type"]),
                "adaptive_support": str(row["adaptive_support"]),
                "remote_support":   str(row["remote_support"]),
            }
            for _, row in batch.iterrows()
        ],
        ids = [str(i) for i in range(start, end)],
    )
    print(f"  Stored {end}/{total} …")

print(f"\n✅  Vector DB saved — {collection.count():,} documents indexed.")


# --------------------------------------------------
# 7) QUICK SMOKE TEST
# --------------------------------------------------

print("\nRunning smoke test...")

test_query = "Python developer with SQL and teamwork skills"
qvec = model.encode([test_query])

results = collection.query(
    query_embeddings=qvec.tolist(),
    n_results=5,
)

print("\nTop 5 Recommended Assessments:\n")
for i, res in enumerate(results["metadatas"][0]):
    print(f"{i+1}. {res['name']}")
    print("   URL      :", res["url"])
    print("   Test Type:", res["test_type"])
    print("-" * 50)

print("\n✅  Embedding pipeline completed successfully!")