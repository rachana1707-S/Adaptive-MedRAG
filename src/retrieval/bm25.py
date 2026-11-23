
import re
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, corpus):
        # Storing the documents and their IDs
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())

        if not self.doc_ids:
            raise ValueError("The corpus is empty")

        # Combining title and text for each document
        self.doc_texts = []

        for doc_id in self.doc_ids:
            document = corpus[doc_id]

            title = document.get("title", "") or ""
            text = document.get("text", "") or ""

            self.doc_texts.append(title + " " + text)

        # Tokenizing the documents before building BM25
        tokenized_corpus = [
            self.tokenize(text)
            for text in self.doc_texts
        ]

        # Creating the BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

    def tokenize(self, text):
        # Converting text to lowercase and splitting into words
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, top_k=5):
        # Checking the number of requested documents
        if top_k <= 0:
            return []

        # Tokenizing the search query
        query_tokens = self.tokenize(query)

        # Getting BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Sorting the documents based on their scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        # Preparing the top retrieved documents
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = self.doc_ids[idx]
            document = self.corpus[doc_id]

            results.append({
                "doc_id": doc_id,
                "chunk_id": None,
                "score": float(scores[idx]),
                "rank": rank,
                "title": document.get("title", ""),
                "text": document.get("text", "")
            })

        return results
