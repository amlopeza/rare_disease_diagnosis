"""Hierarchical text chunking for embedding generation."""

import numpy as np
import pandas as pd


def chunk_text(text, chunk_size=150, overlap=20, max_chunks=15):
    """Split a text into word-level chunks with overlap.

    Parameters
    ----------
    text : str
        Input text to chunk.
    chunk_size : int
        Target number of words per chunk.
    overlap : int
        Number of overlapping words between consecutive chunks.
    max_chunks : int
        Maximum number of chunks to return. If the text produces more,
        the last chunks are dropped.

    Returns
    -------
    list[str]
        List of text chunks. Short texts (<= chunk_size words) return
        a single-element list.
    """
    text = str(text) if text is not None else ""
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    stride = chunk_size - overlap
    chunks = []

    for start in range(0, len(words), stride):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break
        if len(chunks) >= max_chunks:
            break

    return chunks


def chunk_texts(texts, chunk_size=150, overlap=20, max_chunks=15):
    """Chunk a collection of texts and return chunks with a document index.

    Parameters
    ----------
    texts : pd.Series, list, or array-like
        Input texts to chunk.
    chunk_size, overlap, max_chunks : int
        Passed to chunk_text.

    Returns
    -------
    all_chunks : list[str]
        Flat list of all chunks across all documents (ready for embedding).
    doc_lengths : list[int]
        Number of chunks per document. Use this to un-flatten the
        chunk embeddings back to per-document groups.
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    all_chunks = []
    doc_lengths = []

    for text in texts:
        chunks = chunk_text(text, chunk_size=chunk_size,
                            overlap=overlap, max_chunks=max_chunks)
        all_chunks.extend(chunks)
        doc_lengths.append(len(chunks))

    return all_chunks, doc_lengths


def aggregate_chunk_embeddings(chunk_embeddings, doc_lengths, method="mean"):
    """Aggregate chunk-level embeddings into document-level embeddings.

    Parameters
    ----------
    chunk_embeddings : np.ndarray
        Flat matrix of shape (total_chunks, embedding_dim).
    doc_lengths : list[int]
        Number of chunks per document (from chunk_texts).
    method : str
        Aggregation method: "mean", "max", or "mean_max".
        - "mean": average across chunks -> (embedding_dim,)
        - "max": element-wise max across chunks -> (embedding_dim,)
        - "mean_max": concatenation of mean and max -> (2 * embedding_dim,)

    Returns
    -------
    np.ndarray
        Document embeddings, shape (n_docs, dim) where dim is
        embedding_dim for mean/max, or 2*embedding_dim for mean_max.
    """
    doc_embeddings = []
    offset = 0

    for length in doc_lengths:
        doc_chunks = chunk_embeddings[offset:offset + length]
        offset += length

        if method == "mean":
            doc_emb = doc_chunks.mean(axis=0)
        elif method == "max":
            doc_emb = doc_chunks.max(axis=0)
        elif method == "mean_max":
            doc_emb = np.concatenate([
                doc_chunks.mean(axis=0),
                doc_chunks.max(axis=0),
            ])
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean', 'max', or 'mean_max'.")

        doc_embeddings.append(doc_emb)

    return np.stack(doc_embeddings)
