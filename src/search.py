from src.indexer import load_indices


def search_bm25(query: str, top_k: int = 10):
    """BM25 search"""
    inverted_index, bm25, doc_map = load_indices()

    query_tokens = query.lower().split()
    scored_docs = bm25.get_top_n(query_tokens, list(range(len(doc_map))), n=top_k)

    results = []
    for doc_id in scored_docs:
        full_doc = doc_map[doc_id + 1]  # BM25 returns indices, convert to doc_ids
        title, body = full_doc[1], full_doc[2]

        snippet = (body[:150] + "...") if len(body) > 150 else body

        results.append(
            {
                "doc_id": full_doc[0],
                "title": title,
                "snippet": snippet,
                "site": full_doc[3],
                "date": full_doc[4],
            }
        )

    return results
