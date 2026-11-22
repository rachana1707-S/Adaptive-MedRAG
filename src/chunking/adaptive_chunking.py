import re

# Sentence-aware and section-aware adaptive chunking
def adaptive_chunking(doc, tokenizer, nlp, max_tokens=256):

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    text = doc["text"]
    if not text.strip():
        return []

    source = doc["source"]
    chunks = []

    def count_tokens(value):
        return len(tokenizer(
            value,
            add_special_tokens=False,
            truncation=False
        )["input_ids"])

    def add_chunk(start, end, section_index):
        if start >= end or not text[start:end].strip():
            return

        chunk_text = text[start:end]

        chunks.append({
            "doc_id": doc["doc_id"],
            "chunk_id": (
                f"{source}_{doc['doc_id']}"
                f"_adaptive_{max_tokens}_chunk_{len(chunks)}"
            ),
            "text": chunk_text,
            "metadata": {
                "source": source,
                "chunking_method": "adaptive",
                "chunk_size": max_tokens,
                "overlap": 0,
                "section_index": section_index,
                "start_char": start,
                "end_char": end
            }
        })

    # Preserve PubMedQA section positions in original text
    sections = doc.get("sections") if source == "pubmedqa" else None

    if sections:
        section_spans = []
        cursor = 0

        for i, section in enumerate(sections):
            start = text.find(section, cursor)

            if start == -1:
                raise ValueError("Could not locate original section")

            end = start + len(section)
            section_spans.append((start, end, i))
            cursor = end
    else:
        section_spans = [(0, len(text), 0)]

    for sec_start, sec_end, section_index in section_spans:
        section = text[sec_start:sec_end]
        sentences = list(nlp(section).sents)

        current_start = None
        current_end = None

        for sentence in sentences:
            start = sec_start + sentence.start_char
            end = sec_start + sentence.end_char

            if current_start is not None:
                if count_tokens(text[current_start:end]) <= max_tokens:
                    current_end = end
                    continue

                add_chunk(current_start, current_end, section_index)
                current_start = None
                current_end = None

            # Normal sentence fits inside one chunk
            if count_tokens(text[start:end]) <= max_tokens:
                current_start = start
                current_end = end
                continue

            # Oversized sentence: split at whitespace boundaries
            words = list(re.finditer(r"\S+\s*", text[start:end]))
            part_start = start
            part_end = start

            for word in words:
                word_end = start + word.end()

                if count_tokens(text[part_start:word_end]) <= max_tokens:
                    part_end = word_end
                    continue

                if part_end > part_start:
                    add_chunk(part_start, part_end, section_index)
                    part_start = part_end

                # Extremely long single word: fall back to token offsets
                if count_tokens(text[part_start:word_end]) > max_tokens:
                    encoded = tokenizer(
                        text[part_start:word_end],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        truncation=False
                    )
                    offsets = encoded["offset_mapping"]
                    original_start = part_start

                    for token_start in range(0, len(offsets), max_tokens):
                        token_end = min(
                            token_start + max_tokens,
                            len(offsets)
                        )
                        s = original_start + offsets[token_start][0]
                        e = original_start + offsets[token_end - 1][1]
                        add_chunk(s, e, section_index)

                    part_start = word_end
                    part_end = word_end
                else:
                    part_end = word_end

            if part_end > part_start:
                add_chunk(part_start, part_end, section_index)

        if current_start is not None:
            add_chunk(current_start, current_end, section_index)

    return chunks