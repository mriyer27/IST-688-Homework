"""
Run this ONCE (locally, or in your Codespace with network access) to build
the persistent news vector database, BEFORE running or deploying HW7.py.

This keeps embedding costs and latency out of the deployed app itself,
per the assignment's suggestion to build the RAG DB outside the app.

Usage:
    python build_news_db.py

Requires:
    - news.csv in the same folder as this script
    - An OpenAI API key, either as the OPENAI_API_KEY environment variable,
      or you'll be prompted for it
"""

# --- sqlite3 compatibility fix (same reasoning as HW4/HW5) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import csv
import os
import chromadb
from chromadb.utils import embedding_functions

CSV_PATH = "news.csv"
CHROMA_PATH = "./NewsChromaDB"
COLLECTION_NAME = "NewsCollection"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or input("Enter your OpenAI API key: ").strip()


def load_articles(csv_path):
    articles = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            articles.append(
                {
                    "id": f"article_{i}",
                    "company": row["company_name"].strip(),
                    "date": row["Date"],
                    "document": row["Document"],
                    "url": row["URL"],
                }
            )
    return articles


def main():
    articles = load_articles(CSV_PATH)
    print(f"Loaded {len(articles)} articles from {CSV_PATH}")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} documents — skipping rebuild.")
        print(f"Delete the {CHROMA_PATH} folder first if you want to rebuild from scratch.")
        return

    # Embed in batches to avoid oversized requests.
    batch_size = 100
    for start in range(0, len(articles), batch_size):
        batch = articles[start : start + batch_size]
        collection.add(
            ids=[a["id"] for a in batch],
            documents=[a["document"] for a in batch],
            metadatas=[
                {"company": a["company"], "date": a["date"], "url": a["url"]}
                for a in batch
            ],
        )
        print(f"Embedded {min(start + batch_size, len(articles))}/{len(articles)} articles...")

    print(f"Done. Vector DB saved to {CHROMA_PATH} ({collection.count()} documents).")
    print("You can now commit/upload the NewsChromaDB folder alongside HW7.py.")


if __name__ == "__main__":
    main()