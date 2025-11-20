import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.data_loader import get_docs

EMB_DIR = Path("semantic_store")
EMB_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_or_create_embeddings():
    """
    Creates or loads sentence embeddings for all documents.
    Returns:
        embeddings (np.ndarray): shape (N, 384)
        doc_ids (list[int])
        model (SentenceTransformer)
    """
    emb_path = EMB_DIR / "embeddings.npy"
    id_path = EMB_DIR / "doc_ids.npy"

    model = SentenceTransformer(MODEL_NAME)

    if emb_path.exists() and id_path.exists():
        print("Loading semantic embeddings from disk...")
        embeddings = np.load(emb_path)
        doc_ids = np.load(id_path).tolist()
        return embeddings, doc_ids, model

    print("Generating fresh semantic embeddings...")
    docs = get_docs()  # (doc_id, title, body, site, date)
    doc_ids = [d[0] for d in docs]

    texts = [f"{title} {body}" for _, title, body, _, _ in docs]

    embeddings = model.encode(texts, normalize_embeddings=True)
    np.save(emb_path, embeddings)
    np.save(id_path, np.array(doc_ids))

    print(f"Saved {len(embeddings)} embeddings")

    return embeddings, doc_ids, model
