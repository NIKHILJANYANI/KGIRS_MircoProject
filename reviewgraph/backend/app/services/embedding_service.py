"""
embedding_service.py
====================
EmbeddingService for the ReviewGraph project.

Generates sentence embeddings via ``sentence-transformers``, stores them
in a FAISS index for fast approximate nearest-neighbour search, and caches
both the index and the raw embeddings to disk so that re-embedding on
restart is avoided.

Cosine similarity is achieved by L2-normalising all embeddings before
building a FAISS ``IndexFlatIP`` (inner product).  After normalisation,
inner product == cosine similarity.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EMBEDDING_DIM_DEFAULT = 384  # all-MiniLM-L6-v2 output size


class EmbeddingService:
    """
    Manages sentence-level embeddings and FAISS-based similarity search
    for review texts.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/embeddings/faiss.index",
        cache_path: str = "data/embeddings/embedding_cache.npz",
    ) -> None:
        self.model_name  = model_name
        self.index_path  = index_path
        self.cache_path  = cache_path

        self._model      = None          # SentenceTransformer (lazy)
        self._model_ready: bool = False

        self._index: Optional[faiss.Index] = None
        self._id_map: List[str] = []     # position → review_id
        self._embeddings: Optional[np.ndarray] = None  # shape (N, D)

        # Try to restore from existing files
        self.load_index()
        self.load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Lazy-load the SentenceTransformer model.

        Sets ``self._model_ready = False`` and logs a warning if the model
        package or the specific model weights are unavailable (e.g. offline
        environment without pre-downloaded weights).
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._model = SentenceTransformer(self.model_name)
            self._model_ready = True
            logger.info("SentenceTransformer model '%s' loaded.", self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers is not installed.  "
                "Embedding features will be unavailable."
            )
            self._model_ready = False
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer model '%s': %s.  "
                "Embedding features will be unavailable.",
                self.model_name,
                exc,
            )
            self._model_ready = False

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(
        self,
        texts: List[str],
        ids: List[str],
        batch_size: int = 64,
    ) -> Dict:
        """
        Embed *texts*, build a FAISS IndexFlatIP, and persist to disk.

        If a valid cache already exists for the same set of IDs the method
        loads from cache instead of re-embedding.

        Parameters
        ----------
        texts : list[str]
        ids   : list[str]
            Unique review identifiers, parallel to *texts*.
        batch_size : int
            Number of texts embedded per forward pass.

        Returns
        -------
        stats : dict
        """
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length.")

        # ---- Check for valid cache -------------------------------------------
        if self._cache_valid(ids):
            logger.info("Valid embedding cache found; skipping re-embedding.")
            return self.get_stats()

        if not self._model_ready:
            raise RuntimeError(
                "Embedding model not loaded.  Cannot build index."
            )

        logger.info("Building FAISS index for %d texts (batch_size=%d) …", len(texts), batch_size)
        t0 = time.time()

        # ---- Generate embeddings in batches -----------------------------------
        all_embeddings: List[np.ndarray] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch = texts[i * batch_size: (i + 1) * batch_size]
            try:
                emb = self._model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=False,  # we normalise below
                )
                all_embeddings.append(emb.astype(np.float32))
            except Exception as exc:
                logger.error("Embedding batch %d/%d failed: %s", i + 1, n_batches, exc)
                raise

            if (i + 1) % max(1, n_batches // 10) == 0 or (i + 1) == n_batches:
                logger.info(
                    "  Embedded %d / %d texts (%.1f%%).",
                    min((i + 1) * batch_size, len(texts)),
                    len(texts),
                    100 * (i + 1) / n_batches,
                )

        embeddings = np.vstack(all_embeddings)   # (N, D)

        # ---- L2-normalise for cosine similarity via inner product -------------
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # ---- Build FAISS index ------------------------------------------------
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        elapsed = time.time() - t0
        logger.info("FAISS index built in %.2f s (%d vectors, dim=%d).", elapsed, len(ids), dim)

        # ---- Persist -----------------------------------------------------------
        self._persist(index, ids, embeddings)

        self._index      = index
        self._id_map     = list(ids)
        self._embeddings = embeddings

        return {
            "n_vectors":  len(ids),
            "dim":        dim,
            "index_type": "IndexFlatIP (cosine via L2-norm)",
            "build_time_seconds": round(elapsed, 3),
            "index_path": self.index_path,
            "cache_path": self.cache_path,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_text: str, k: int = 10) -> List[Dict]:
        """
        Find the *k* most similar reviews to *query_text*.

        Returns
        -------
        list[dict] with keys ``review_id`` and ``score`` (cosine similarity).
        """
        if not self.is_ready():
            logger.warning("Index not ready; returning empty results.")
            return []

        if not self._model_ready:
            raise RuntimeError("Embedding model not loaded; cannot encode query.")

        q_emb = self._model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        return self._run_search(q_emb, k)

    def search_by_id(self, review_id: str, k: int = 10) -> List[Dict]:
        """
        Find the *k* most similar reviews to the review identified by *review_id*.

        The embedding is looked up from the cached array rather than
        re-encoding the original text.

        Returns
        -------
        list[dict] with keys ``review_id`` and ``score``.
        """
        if not self.is_ready():
            logger.warning("Index not ready; returning empty results.")
            return []

        if review_id not in self._id_map:
            logger.warning("review_id '%s' not found in index.", review_id)
            return []

        idx   = self._id_map.index(review_id)
        q_emb = self._embeddings[[idx]].astype(np.float32)

        results = self._run_search(q_emb, k + 1)   # +1 to exclude the query itself
        return [r for r in results if r["review_id"] != review_id][:k]

    def _run_search(self, q_emb: np.ndarray, k: int) -> List[Dict]:
        """Execute a FAISS search and package results."""
        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            return []

        try:
            scores, indices = self._index.search(q_emb, actual_k)
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return []

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS sentinel for "not enough results"
                continue
            results.append({
                "review_id": self._id_map[int(idx)],
                "score":     round(float(score), 6),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        index: faiss.Index,
        ids: List[str],
        embeddings: np.ndarray,
    ) -> None:
        """Write FAISS index and embedding cache to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, self.index_path)
        np.savez_compressed(
            self.cache_path,
            ids=np.array(ids, dtype=object),
            embeddings=embeddings,
        )
        logger.info("Saved FAISS index → '%s'.", self.index_path)
        logger.info("Saved embedding cache → '%s'.", self.cache_path)

    def load_index(self) -> None:
        """
        Restore FAISS index and id mapping from disk.
        Supports both .npz compressed cache and .pkl dictionary cache formats.
        """
        import pickle

        models_dir = os.path.dirname(self.index_path) or "models"
        idx_candidates = [
            self.index_path,
            os.path.join(models_dir, "review_embeddings.faiss"),
            "models/review_embeddings.faiss",
            "../models/review_embeddings.faiss",
        ]
        chosen_idx = next((p for p in idx_candidates if os.path.exists(p)), None)

        cache_candidates = [
            self.cache_path,
            os.path.join(models_dir, "review_embeddings_cache.pkl"),
            "models/review_embeddings_cache.pkl",
            "../models/review_embeddings_cache.pkl",
            os.path.join(models_dir, "review_embeddings_cache.npz"),
            "models/embeddings_cache.npz",
            "../models/embeddings_cache.npz",
        ]
        chosen_cache = next((p for p in cache_candidates if os.path.exists(p)), None)

        if not (chosen_idx and chosen_cache):
            logger.info("No existing FAISS index / cache found; will build on first use.")
            return

        try:
            self._index = faiss.read_index(chosen_idx)
            if chosen_cache.endswith(".pkl"):
                with open(chosen_cache, "rb") as fh:
                    cdata = pickle.load(fh)
                if isinstance(cdata, dict):
                    self._embeddings = cdata.get("embeddings")
                    self._id_map = list(cdata.get("review_ids", []))
            else:
                data = np.load(chosen_cache, allow_pickle=True)
                self._id_map = list(data["ids"])
                self._embeddings = data["embeddings"].astype(np.float32)

            # If id_map is still empty, look for review_id_map.pkl
            if not self._id_map:
                id_map_p = os.path.join(models_dir, "review_id_map.pkl")
                if os.path.exists(id_map_p):
                    with open(id_map_p, "rb") as fh:
                        raw_id_map = pickle.load(fh)
                    if isinstance(raw_id_map, dict):
                        self._id_map = [raw_id_map[i] for i in range(len(raw_id_map))]

            logger.info(
                "Loaded FAISS index (%d vectors, dim=%d) from '%s'.",
                self._index.ntotal,
                self._embeddings.shape[1] if self._embeddings is not None else 0,
                chosen_idx,
            )
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s. Index will be rebuilt.", exc)
            self._index      = None
            self._id_map     = []
            self._embeddings = None

    # ------------------------------------------------------------------
    # Cache validation
    # ------------------------------------------------------------------

    def _cache_valid(self, ids: List[str]) -> bool:
        """
        Return True iff the on-disk cache exists and covers exactly the
        same set of IDs as *ids* (order-independent).
        """
        if self._embeddings is None or self._index is None:
            return False
        if len(self._id_map) != len(ids):
            return False
        return set(self._id_map) == set(ids)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if the FAISS index is loaded and non-empty."""
        return (
            self._index is not None
            and self._index.ntotal > 0
            and len(self._id_map) > 0
            and self._embeddings is not None
        )

    def get_stats(self) -> Dict:
        """Return a statistics dict describing the current index state."""
        return {
            "index_ready":     self.is_ready(),
            "model_ready":     self._model_ready,
            "model_name":      self.model_name,
            "n_vectors":       self._index.ntotal if self._index else 0,
            "embedding_dim":   int(self._embeddings.shape[1])
                               if self._embeddings is not None and len(self._embeddings.shape) > 1
                               else _EMBEDDING_DIM_DEFAULT,
            "index_path":      self.index_path,
            "cache_path":      self.cache_path,
            "index_type":      "IndexFlatIP (cosine via L2-norm)",
        }
