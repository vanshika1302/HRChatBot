"""
ingest.py

Reads the HR policy document, splits it into overlapping text chunks,
embeds each chunk with a local sentence-transformers model, and persists
the chunk texts + embeddings to disk so chatbot.py can retrieve from them.

Run this once (and again any time the source document changes) before
starting the app:

    python ingest.py [--input hr_policy.txt] [--output-dir vector_data]
"""

import argparse
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "hr_policy.txt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "vector_data")


def chunk_text(text, chunk_size=600, chunk_overlap=100):
    """Split text into overlapping chunks of roughly `chunk_size` characters.

    A small, dependency-free character-window splitter. Good enough for
    short policy documents; swap in something fancier if the source
    documents get large or structured.
    """
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    length = len(text)
    step = chunk_size - chunk_overlap

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start += step

    return chunks


def ingest(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT_DIR,
           model_name=DEFAULT_MODEL_NAME, chunk_size=600, chunk_overlap=100):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    texts = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not texts:
        raise ValueError(f"No text found in {input_path}")

    print(f"Loaded {input_path}; split into {len(texts)} chunk(s).")

    print(f"Loading embedding model '{model_name}' ...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    os.makedirs(output_dir, exist_ok=True)
    texts_path = os.path.join(output_dir, "texts.pkl")
    embeddings_path = os.path.join(output_dir, "embeddings.npy")

    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
    np.save(embeddings_path, embeddings)

    print(f"Ingested {len(texts)} chunk(s). Embeddings saved to {output_dir}/")
    return texts, embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk and embed the HR policy document for retrieval."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Path to the source text file (default: hr_policy.txt)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the embeddings/texts (default: vector_data/)",
    )
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL_NAME,
        help="sentence-transformers model to use for embeddings",
    )
    parser.add_argument("--chunk-size", type=int, default=600, help="Max characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Character overlap between consecutive chunks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(
        input_path=args.input,
        output_dir=args.output_dir,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
