# Split a document into fixed-size token windows
def fixed_size_chunking(doc, tokenizer, chunk_size=256, overlap=50):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size")

    text = doc["text"]

    if not text.strip():
        return []

    # Get token offsets so we can preserve the original text
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    offsets = encoded["offset_mapping"]
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(offsets), step):
        end = min(start + chunk_size, len(offsets))

        # Find the character positions of this token window
        start_char = offsets[start][0]
        end_char = offsets[end - 1][1]

        chunk_text = text[start_char:end_char]

        chunks.append({
            "doc_id": doc["doc_id"],
            "chunk_id": f"{doc['source']}_{doc['doc_id']}_fixed_{chunk_size}_{overlap}_chunk_{len(chunks)}",
            "text": chunk_text,
            "metadata": {
                "source": doc["source"],
                "chunking_method": "fixed",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "start_token": start,
                "end_token": end,
                "start_char": start_char,
                "end_char": end_char
            }
        })

        if end == len(offsets):
            break

    return chunks