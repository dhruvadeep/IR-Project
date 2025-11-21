from src.indexer import load_indices
from src.semantic.vector_store import SemanticVectorStore

# Load once (lazy initialization)
store = None


def semantic_search(query: str, top_k: int = 10, include_text: bool = False):
    global store
    if store is None:
        store = SemanticVectorStore()

    # Compute similarity
    results = store.cosine_top_k(store.query_to_vector(query), k=top_k)

    # Load doc_map to return metadata
    _, _, doc_map = load_indices()

    final = []
    for doc_id, score in results:
        d = doc_map[doc_id]
        title, body, site, date = d[1], d[2], d[3], d[4]
        snippet = (body[:150] + "...") if len(body) > 150 else body

        item = {
            "doc_id": doc_id,
            "title": title,
            "snippet": snippet,
            "site": site,
            "date": date,
            "score": score,
        }
        if include_text:
            item["text"] = body

        final.append(item)

    return final
