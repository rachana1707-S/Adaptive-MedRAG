
class HybridRetriever:

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        rrf_k=60,
        candidate_k=100
    ):
        # Storing the two retrieval models
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

        # Parameters used for rank fusion
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        # Both retrievers should use the same documents
        if set(self.bm25.corpus) != set(self.dense.corpus):
            raise ValueError(
                "BM25 and Dense retrievers must use the same corpus"
            )

    def reciprocal_rank_fusion(self, bm25_results, dense_results):
        # Combining the rankings using RRF scores
        rrf_scores = {}

        for results in [bm25_results, dense_results]:

            for rank, result in enumerate(results, start=1):
                doc_id = result["doc_id"]

                # Adding the contribution from this ranking
                rrf_scores[doc_id] = (
                    rrf_scores.get(doc_id, 0)
                    + 1 / (self.rrf_k + rank)
                )

        # Sorting documents by their combined RRF scores
        ranked_docs = sorted(
            rrf_scores.items(),
            key=lambda item: (-item[1], str(item[0]))
        )

        return ranked_docs

    def search(self, query, top_k=5):
        # Checking the requested number of results
        if top_k <= 0:
            return []

        # Retrieving candidates using both methods
        candidate_count = max(top_k, self.candidate_k)

        bm25_results = self.bm25.search(
            query,
            top_k=candidate_count
        )

        dense_results = self.dense.search(
            query,
            top_k=candidate_count
        )

        # Combining the two ranked lists
        fused_results = self.reciprocal_rank_fusion(
            bm25_results,
            dense_results
        )

        results = []

        # Using the shared corpus to get document information
        corpus = self.bm25.corpus

        for rank, (doc_id, score) in enumerate(
            fused_results[:top_k],
            start=1
        ):
            document = corpus[doc_id]

            results.append({
                "doc_id": doc_id,
                "chunk_id": None,
                "score": float(score),
                "rank": rank,
                "title": document.get("title", ""),
                "text": document.get("text", "")
            })

        return results
