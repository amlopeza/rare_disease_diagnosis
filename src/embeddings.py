#%% Imports
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


#%% Load model
def load_model(model_name="FremyCompany/BioLORD-2023"):
    """Load a SentenceTransformer model."""
    return SentenceTransformer(model_name)


#%% Generate embeddings
def generate_embeddings(texts, model=None, model_name="FremyCompany/BioLORD-2023",
                        batch_size=64, show_progress=True):
    """Generate sentence embeddings for a collection of texts.

    Parameters
    ----------
    texts : pd.Series, list, or array-like
        Input texts to encode.
    model : SentenceTransformer, optional
        Pre-loaded model. If None, loads using model_name.
    model_name : str
        Model to load if model is not provided.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_texts, embedding_dim).
    """
    if model is None:
        model = load_model(model_name)

    # Ensure texts is a list of strings
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    texts = [str(t) if t is not None else "" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )

    return np.array(embeddings)


#%% MedCPT models
def load_medcpt(encoder_type="article"):
    """Load MedCPT query or article encoder.

    Parameters
    ----------
    encoder_type : str
        "query" for encoding test/search queries (max 64 tokens),
        "article" for encoding train/reference documents (max 512 tokens).

    Returns
    -------
    tuple
        (tokenizer, model, max_length)
    """
    if encoder_type == "query":
        name = "ncbi/MedCPT-Query-Encoder"
        max_length = 64
    else:
        name = "ncbi/MedCPT-Article-Encoder"
        max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tokenizer, model, max_length


def generate_embeddings_medcpt(texts, tokenizer, model, max_length,
                               batch_size=64, normalize=True,
                               show_progress=True):
    """Generate embeddings using MedCPT (CLS token pooling).

    Parameters
    ----------
    texts : pd.Series, list, or array-like
        Input texts to encode.
    tokenizer : AutoTokenizer
        MedCPT tokenizer.
    model : AutoModel
        MedCPT encoder (query or article).
    max_length : int
        Max token length (64 for query, 512 for article).
    batch_size : int
        Batch size for encoding.
    normalize : bool
        Whether to L2-normalize embeddings.
    show_progress : bool
        Whether to print progress.

    Returns
    -------
    np.ndarray
        Matrix of shape (n_texts, 768).
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [str(t) if t is not None else "" for t in texts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            embeds = model(**encoded).last_hidden_state[:, 0, :]

        all_embeddings.append(embeds.cpu().numpy())

        if show_progress:
            batch_num = i // batch_size + 1
            print(f"\r  Batch {batch_num}/{n_batches}", end="", flush=True)

    if show_progress:
        print()

    embeddings = np.concatenate(all_embeddings, axis=0)

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

    return embeddings
