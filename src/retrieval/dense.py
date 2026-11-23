
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(
        self,
        corpus,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32
    ):
        # Storing the corpus and document IDs
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())

        if not self.doc_ids:
            raise ValueError("The corpus is empty")

        self.model_name = model_name

        # Loading the Sentence Transformer model
        self.model = SentenceTransformer(model_name)

        # Combining the title and text of each document
        self.doc_texts = []

        for doc_id in self.doc_ids:
            document = corpus[doc_id]

            title = document.get("title", "") or ""
            text = document.get("text", "") or ""

            self.doc_texts.append(title + " " + text)

        # Generating embeddings for all documents
        # We normalize them for cosine similarity
        self.document_embeddings = self.model.encode_document(
            self.doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def search(self, query, top_k=5):
        # Checking the requested number of results
        if top_k <= 0:
            return []

        # Creating an embedding for the query
        query_embedding = self.model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Comparing the query with the document embeddings
        # Dot product gives cosine similarity for normalized vectors
        scores = self.document_embeddings @ query_embedding

        # Sorting documents by similarity score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

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
