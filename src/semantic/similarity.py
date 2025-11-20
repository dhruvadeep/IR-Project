from src.semantic.vector_store import SemanticVectorStore
from src.indexer import load_indices

store = None


def similar_articles(doc_id: int, top_k: int = 5):
    global store
    if store is None:
        store = SemanticVectorStore()

    results = store.similar_to_doc(doc_id, k=top_k)

    # Load doc metadata
    _, _, doc_map = load_indices()

    final = []
    for sim_id, score in results:
        d = doc_map[sim_id]
        title, body, site, date = d[1], d[2], d[3], d[4]
        snippet = (body[:150] + "...") if len(body) > 150 else body

        final.append(
            {
                "doc_id": sim_id,
                "title": title,
                "snippet": snippet,
                "site": site,
                "date": date,
                "similarity": score,
            }
        )

    return final
