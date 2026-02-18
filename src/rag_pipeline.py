"""
rag_pipeline.py
─────────────────────────────────────────────────────────────────
RAG pipeline — Python 3.10 · Windows · 8 GB RAM

Embedding model: all-MiniLM-L6-v2 (22 MB, 384-dim, ~14 ms/sentence CPU)
Vector store   : FAISS IndexFlatIP  (cosine on unit vectors, sub-ms retrieval)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from loguru import logger


@dataclass
class ActionMatch:
    """A retrieved action candidate with its confidence score."""
    action_id:   str
    name:        str
    description: str
    category:    str
    parameters:  dict
    executor:    str
    destructive: bool
    confirmation_required: bool
    confidence:  float          # cosine similarity [0..1]
    matched_example: str = ""   # which example phrase triggered the match


class RAGPipeline:
    """
    Offline RAG pipeline backed by FAISS and sentence-transformers.

    Parameters
    ----------
    actions_path      : path to actions.json
    embedding_model   : huggingface model name (auto-downloaded once)
    index_path        : where to persist FAISS index
    metadata_path     : where to persist action metadata mapping
    top_k             : how many candidates to return
    confidence_threshold : minimum score to return a match
    """

    def __init__(
        self,
        actions_path:         str = "data/actions/actions.json",
        embedding_model:      str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path:           str = "models/faiss/actions.index",
        metadata_path:        str = "models/faiss/metadata.json",
        top_k:                int = 3,
        confidence_threshold: float = 0.55,
    ):
        self.actions_path         = Path(actions_path)
        self.embedding_model_name = embedding_model
        self.index_path           = Path(index_path)
        self.metadata_path        = Path(metadata_path)
        self.top_k                = top_k
        self.confidence_threshold = confidence_threshold

        self._embedder = None    # lazy-loaded SentenceTransformer
        self._index    = None    # FAISS index
        self._meta: List[dict] = []   # parallel list to index rows

    # ── Setup ────────────────────────────────────────────────────────────────
    def setup(self, force_rebuild: bool = False):
        """
        Load or build the FAISS index.

        If a pre-built index exists on disk, it is loaded instantly.
        Otherwise (or if force_rebuild=True), we embed all examples and save.
        """
        self._load_embedder()

        if (
            not force_rebuild
            and self.index_path.exists()
            and self.metadata_path.exists()
        ):
            self._load_index()
        else:
            self._build_index()

    def _load_embedder(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        logger.info(f"[rag] Loading embedder: {self.embedding_model_name} …")
        t0 = time.perf_counter()
        self._embedder = SentenceTransformer(self.embedding_model_name)
        logger.info(
            f"[rag] Embedder loaded in {(time.perf_counter()-t0)*1000:.0f} ms"
        )

    def _load_index(self):
        import faiss
        logger.info("[rag] Loading pre-built FAISS index from disk …")
        self._index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path) as f:
            self._meta = json.load(f)
        logger.info(
            f"[rag] Index ready: {self._index.ntotal} vectors, "
            f"{len(set(m['action_id'] for m in self._meta))} actions"
        )

    def _build_index(self):
        """Embed all example phrases and build the FAISS index."""
        import faiss

        logger.info("[rag] Building FAISS index …")
        t0 = time.perf_counter()

        with open(self.actions_path) as f:
            actions = json.load(f)

        phrases: List[str] = []
        meta:    List[dict] = []

        for action in actions:
            for example in action["examples"]:
                phrases.append(example)
                meta.append({
                    "action_id":            action["id"],
                    "name":                 action["name"],
                    "description":          action["description"],
                    "category":             action["category"],
                    "parameters":           action["parameters"],
                    "executor":             action["executor"],
                    "destructive":          action["destructive"],
                    "confirmation_required":action["confirmation_required"],
                    "example":              example,
                })

        logger.info(f"[rag] Embedding {len(phrases)} example phrases …")
        embeddings = self._embedder.encode(
            phrases,
            batch_size    = 64,
            show_progress_bar = False,
            normalize_embeddings = True,    # unit vectors → dot product = cosine
        ).astype(np.float32)

        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)      # Inner Product on unit vectors = cosine
        index.add(embeddings)

        # Persist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

        self._index = index
        self._meta  = meta

        elapsed = (time.perf_counter() - t0) * 1_000
        logger.info(
            f"[rag] Index built in {elapsed:.0f} ms | "
            f"{len(phrases)} vectors, dim={dim}"
        )

    # ── Query ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> List[ActionMatch]:
        """
        Retrieve the top-K most relevant actions for a transcribed query.

        Returns a list of ActionMatch sorted by confidence descending.
        Only results above confidence_threshold are returned.
        """
        if self._index is None:
            raise RuntimeError("Call setup() before retrieve()")

        t0 = time.perf_counter()

        # Embed query (normalised for cosine similarity)
        q_vec = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Search FAISS
        scores, indices = self._index.search(q_vec, self.top_k * 3)
        # top_k * 3 so we can de-duplicate by action_id and still get top_k

        # De-duplicate: keep only best score per action_id
        seen: Dict[str, ActionMatch] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            m   = self._meta[idx]
            aid = m["action_id"]
            if aid not in seen or score > seen[aid].confidence:
                seen[aid] = ActionMatch(
                    action_id              = aid,
                    name                   = m["name"],
                    description            = m["description"],
                    category               = m["category"],
                    parameters             = m["parameters"],
                    executor               = m["executor"],
                    destructive            = m["destructive"],
                    confirmation_required  = m["confirmation_required"],
                    confidence             = float(score),
                    matched_example        = m["example"],
                )

        # Filter and sort
        matches = sorted(seen.values(), key=lambda x: x.confidence, reverse=True)
        matches = [m for m in matches if m.confidence >= self.confidence_threshold]
        matches = matches[: self.top_k]

        elapsed = (time.perf_counter() - t0) * 1_000
        logger.debug(
            f"[rag] Retrieved {len(matches)} actions for '{query}' "
            f"in {elapsed:.1f} ms"
        )

        return matches

    # ── Convenience ──────────────────────────────────────────────────────────
    def best_match(self, query: str) -> Optional[ActionMatch]:
        """Return the single best matching action, or None if below threshold."""
        results = self.retrieve(query)
        return results[0] if results else None
