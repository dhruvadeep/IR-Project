import math
from typing import Dict, List, Set


def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Precision@K.
    Precision = (Relevant Retrieved) / (Total Retrieved)
    """
    if k <= 0:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)

    return relevant_retrieved / k


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Recall@K.
    Recall = (Relevant Retrieved) / (Total Relevant)
    """
    if not relevant_ids:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)

    return relevant_retrieved / len(relevant_ids)


def average_precision(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """
    Calculate Average Precision (AP) for a single query.
    """
    if not relevant_ids:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_hits += 1
            score += num_hits / (i + 1.0)

    return score / len(relevant_ids)


def mean_average_precision(
    results: Dict[str, List[int]], ground_truth: Dict[str, Set[int]]
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.
    """
    total_ap = 0.0
    num_queries = len(results)

    if num_queries == 0:
        return 0.0

    for query, retrieved_ids in results.items():
        relevant_ids = ground_truth.get(query, set())
        total_ap += average_precision(retrieved_ids, relevant_ids)

    return total_ap / num_queries


def dcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG) at K.
    Binary relevance assumption: 1 if relevant, 0 otherwise.
    """
    dcg = 0.0
    for i in range(min(len(retrieved_ids), k)):
        doc_id = retrieved_ids[i]
        rel = 1 if doc_id in relevant_ids else 0
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Calculate Normalized DCG (nDCG) at K.
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)

    # Calculate Ideal DCG (IDCG)
    # IDCG is the DCG of the perfect ordering (all relevant docs at the top)
    ideal_retrieved = sorted(
        list(relevant_ids), key=lambda x: 1, reverse=True
    )  # Just need count
    # Actually, for binary relevance, IDCG is just filling the top positions with 1s
    num_relevant = len(relevant_ids)
    idcg = 0.0
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_system(
    results: Dict[str, List[int]], ground_truth: Dict[str, Set[int]], k: int = 10
) -> Dict[str, float]:
    """
    Run full evaluation suite.
    """
    precisions = []
    recalls = []
    ndcgs = []

    for query, retrieved_ids in results.items():
        relevant_ids = ground_truth.get(query, set())
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    return {
        "MAP": mean_average_precision(results, ground_truth),
        f"Mean Precision@{k}": sum(precisions) / len(precisions) if precisions else 0.0,
        f"Mean Recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"Mean nDCG@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }
