
import numpy as np


def recall_at_k(retrieved_ids, relevant_docs, k):
    # Checking how many relevant documents were retrieved
    if k <= 0:
        return 0.0

    relevant_ids = {
        doc_id
        for doc_id, score in relevant_docs.items()
        if score > 0
    }

    if not relevant_ids:
        return 0.0

    # Getting the top k retrieved document IDs
    top_ids = set(retrieved_ids[:k])

    # Counting the relevant documents found
    hits = len(top_ids.intersection(relevant_ids))

    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevant_docs, k):
    # Measuring how well the relevant documents are ranked
    if k <= 0:
        return 0.0

    dcg = 0.0

    # Calculating DCG for our retrieved ranking
    for rank, doc_id in enumerate(retrieved_ids[:k]):
        relevance = max(0, relevant_docs.get(doc_id, 0))

        dcg += (
            (2 ** relevance - 1)
            / np.log2(rank + 2)
        )

    # Calculating DCG for the ideal ranking
    ideal_scores = sorted(
        [max(0, score) for score in relevant_docs.values()],
        reverse=True
    )[:k]

    idcg = 0.0

    for rank, relevance in enumerate(ideal_scores):
        idcg += (
            (2 ** relevance - 1)
            / np.log2(rank + 2)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_retriever(retriever, queries, qrels, top_k=10):
    # Evaluating a retrieval method on the test queries
    import time
    import pandas as pd

    evaluation_results = []

    for query_id, relevant_docs in qrels.items():
        if query_id not in queries:
            continue

        question = queries[query_id]

        # Measuring the time taken for searching
        start_time = time.perf_counter()

        results = retriever.search(question, top_k=top_k)

        search_time = (
            time.perf_counter() - start_time
        ) * 1000

        # Getting document IDs in ranked order
        retrieved_ids = [
            result["doc_id"]
            for result in results
        ]

        evaluation_results.append({
            "query_id": query_id,
            "recall_5": recall_at_k(
                retrieved_ids, relevant_docs, 5
            ),
            "recall_10": recall_at_k(
                retrieved_ids, relevant_docs, 10
            ),
            "ndcg_10": ndcg_at_k(
                retrieved_ids, relevant_docs, 10
            ),
            "search_time_ms": search_time
        })

    return pd.DataFrame(evaluation_results)
