"""
retriever.py — FAISS-based FAQ retriever.
Loads your local models/faiss/faq_index.faiss + faq_metadata.pkl
and finds the closest answer for any informative query.
"""

import logging
import os
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Paths — adjust if your folder layout differs ──────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # project root
FAISS_INDEX = BASE_DIR / "models" / "faiss" / "faq_index.faiss"
METADATA    = BASE_DIR / "models" / "faiss" / "faq_metadata.pkl"

# Must match the model used when you built the index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many nearest neighbours to retrieve (we use only the top-1 answer)
TOP_K = 3

# Similarity threshold: cosine distance above this → "I don't know"
# (FAISS L2 distance; lower = more similar. ~0.8 is a safe cutoff for MiniLM)
MAX_DISTANCE = 0.8


class FAISSRetriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX,
        metadata_path: Path = METADATA,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        logger.info(f"Loading FAISS index from {index_path} ...")
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Make sure models/faiss/faq_index.faiss exists."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Make sure models/faiss/faq_metadata.pkl exists."
            )

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            raw = pickle.load(f)

        # Normalise to list-of-dicts regardless of how the pickle was built.
        # Supported formats:
        #   A) list of strings   → ["answer text", ...]           (your current format)
        #   B) list of dicts     → [{"question":..,"answer":..}]  (extended format)
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], str):
                # Format A — only answers, no question text stored
                self.metadata = [{"question": "", "answer": a} for a in raw]
                logger.info("Metadata format: list of strings (answers only)")
            elif isinstance(raw[0], dict):
                # Format B — already has question/answer keys
                self.metadata = raw
                logger.info("Metadata format: list of dicts")
            else:
                raise ValueError(f"Unexpected metadata item type: {type(raw[0])}")
        else:
            raise ValueError(f"Unexpected metadata format: {type(raw)}")

        logger.info(f"Loading sentence-transformer: {embedding_model} ...")
        self.embedder = SentenceTransformer(embedding_model)

        logger.info(
            f"FAISS retriever ready — {self.index.ntotal} vectors indexed"
        )

    def search(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Returns:
            {
                "answer":    str,
                "question":  str,   # the matched FAQ question
                "distance":  float,
                "confident": bool   # False when distance > MAX_DISTANCE
            }
        """
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        distances, indices = self.index.search(query_vec, top_k)

        best_dist = float(distances[0][0])
        best_idx  = int(indices[0][0])

        if best_idx == -1 or best_dist > MAX_DISTANCE:
            return {
                "answer":    (
                    "I'm sorry, I couldn't find a specific answer to your question. "
                    "Please contact our support team for further assistance."
                ),
                "question":  "",
                "distance":  best_dist,
                "confident": False,
            }

        match = self.metadata[best_idx]
        return {
            "answer":    match.get("answer", "No answer available."),
            "question":  match.get("question", ""),
            "distance":  round(best_dist, 4),
            "confident": True,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever