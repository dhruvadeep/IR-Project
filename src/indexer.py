import pickle
import re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.data_loader import get_docs

INDICES_DIR = Path("indices")
INDICES_DIR.mkdir(exist_ok=True)


def tokenize(text):
    """Simple tokenizer"""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_indices():
    """Build and persist inverted index + BM25"""
    print("Fetching docs from DB...")
    docs = get_docs()  # (doc_id, title, body, site, date)

    print(f"Building indices for {len(docs)} docs...")

    # Map doc_id to doc content
    doc_map = {doc[0]: doc for doc in docs}  # doc_id -> full record

    # Inverted index: word -> set of doc_ids
    inverted_index = defaultdict(set)
    documents_text = []  # For BM25

    for doc_id, title, body, site, date in docs:
        text = f"{title} {body}"
        documents_text.append(text)

        words = tokenize(text)
        for word in set(words):
            inverted_index[word].add(doc_id)

    # Build BM25
    tokenized_docs = [tokenize(text) for text in documents_text]
    bm25 = BM25Okapi(tokenized_docs)

    # Persist
    with open(INDICES_DIR / "inverted_index.pkl", "wb") as f:
        pickle.dump(dict(inverted_index), f)

    with open(INDICES_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(INDICES_DIR / "doc_map.pkl", "wb") as f:
        pickle.dump(doc_map, f)

    print(f"✓ Indices saved. Vocab size: {len(inverted_index)}")


def load_indices():
    """Load persisted indices"""
    with open(INDICES_DIR / "inverted_index.pkl", "rb") as f:
        inverted_index = pickle.load(f)

    with open(INDICES_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(INDICES_DIR / "doc_map.pkl", "rb") as f:
        doc_map = pickle.load(f)

    return inverted_index, bm25, doc_map
