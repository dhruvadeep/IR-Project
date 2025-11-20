import numpy as np
from src.semantic.encoder import load_or_create_embeddings


class SemanticVectorStore:
    def __init__(self):
        self.embeddings, self.doc_ids, self.model = load_or_create_embeddings()

        # Map: doc_id -> index in embedding matrix
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def query_to_vector(self, query: str):
        return self.model.encode([query], normalize_embeddings=True)[0]

    def cosine_top_k(self, vector, k=10):
        scores = self.embeddings @ vector
        top_idx = np.argsort(scores)[::-1][:k]

        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def similar_to_doc(self, doc_id, k=5):
        if doc_id not in self.id_to_index:
            raise ValueError(f"doc_id {doc_id} not found")

        idx = self.id_to_index[doc_id]
        target_vec = self.embeddings[idx]

        scores = self.embeddings @ target_vec
        scores[idx] = -1  # exclude itself

        top_idx = np.argsort(scores)[::-1][:k]

        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]
