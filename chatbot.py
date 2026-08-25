"""
chatbot.py

Retrieval-augmented Q&A over the HR policy document, using only local,
free models - no API keys, no external services:

- sentence-transformers ("all-MiniLM-L6-v2") for embeddings, with
  cosine-similarity retrieval done by hand via numpy/scipy (no vector DB)
- google/flan-t5-small for grounded answer generation

Run `ingest.py` first to build the vector_data/ index this module reads.
"""

import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VECTOR_DIR = os.path.join(BASE_DIR, "vector_data")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"

# Lazily-initialized singletons. Importing this module (or parsing CLI args
# in app.py) should never trigger a model download - only answering a
# question should.
_embed_model = None
_generator = None
_texts = None
_embeddings = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_generator():
    global _generator
    if _generator is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1
        _generator = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer, device=device
        )
    return _generator


def load_index(vector_dir=DEFAULT_VECTOR_DIR):
    """Load the persisted chunk texts + embeddings produced by ingest.py."""
    global _texts, _embeddings
    texts_path = os.path.join(vector_dir, "texts.pkl")
    embeddings_path = os.path.join(vector_dir, "embeddings.npy")

    if not os.path.exists(texts_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"No index found in '{vector_dir}/'. Run `python ingest.py` first."
        )

    with open(texts_path, "rb") as f:
        _texts = pickle.load(f)
    _embeddings = np.load(embeddings_path)
    return _texts, _embeddings


def _ensure_index_loaded(vector_dir=DEFAULT_VECTOR_DIR):
    if _texts is None or _embeddings is None:
        load_index(vector_dir)
    return _texts, _embeddings


def retrieve_top_k(question, k=2, vector_dir=DEFAULT_VECTOR_DIR):
    """Embed the question and return the top-k most similar chunks (cosine similarity)."""
    texts, embeddings = _ensure_index_loaded(vector_dir)
    k = max(1, min(k, len(texts)))

    embed_model = _get_embed_model()
    q_emb = embed_model.encode([question], convert_to_numpy=True)  # shape (1, dim)
    distances = cdist(q_emb, embeddings, metric="cosine")[0]  # shape (n_chunks,)
    topk_idx = np.argsort(distances)[:k]
    top_texts = [texts[i] for i in topk_idx]
    top_scores = [1 - float(distances[i]) for i in topk_idx]  # distance -> similarity
    return top_texts, top_scores


def build_prompt(contexts, question):
    prompt = (
        "You are an assistant that answers questions using ONLY the provided "
        "context. If the answer is not present, say "
        "\"I don't know based on the provided documents.\"\n\n"
    )
    for i, c in enumerate(contexts, 1):
        prompt += f"[Context {i}]\n{c}\n\n"
    prompt += f"Question: {question}\nAnswer concisely and cite the context like [Context 1]."
    return prompt


def answer_question(question, k=2, max_new_tokens=256, vector_dir=DEFAULT_VECTOR_DIR):
    """Retrieve relevant chunks and generate an answer grounded in them."""
    contexts, scores = retrieve_top_k(question, k=k, vector_dir=vector_dir)
    prompt = build_prompt(contexts, question)

    generator = _get_generator()
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    answer = out[0]["generated_text"]

    return {
        "answer": answer,
        "sources": [
            {"text": contexts[i], "score": scores[i], "idx": i}
            for i in range(len(contexts))
        ],
    }


if __name__ == "__main__":
    # Small manual smoke test: `python chatbot.py` after running ingest.py.
    q = "When is payroll processed?"
    result = answer_question(q)
    print("Q:", q)
    print("A:", result["answer"])
    print("Sources:", result["sources"])
