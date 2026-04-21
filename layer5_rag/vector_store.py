 
from __future__ import annotations
 
import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
 
import numpy as np
from sentence_transformers import SentenceTransformer
 
from playbook_kb import get_all_chunks
from schemas_layer5 import PlaybookChunk, RetrievedChunk
 
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
FAISS_PATH = DATA_DIR / "faiss_index.pkl"
 
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
COLLECTION_NAME  = "supply_chain_playbooks"
CHUNK_SIZE_TOKENS = 512
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING HELPER
# ─────────────────────────────────────────────────────────────────────────────
 
class EmbeddingEngine:
    """Singleton wrapper around sentence-transformers model."""
    _model: Optional[SentenceTransformer] = None
 
    @classmethod
    def get(cls) -> SentenceTransformer:
        if cls._model is None:
            print(f"  [↓] Loading embedding model ({EMBEDDING_MODEL}) …")
            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._model
 
    @classmethod
    def encode(cls, texts: List[str]) -> np.ndarray:
        return cls.get().encode(texts, show_progress_bar=False, normalize_embeddings=True)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CHROMA BACKEND
# ─────────────────────────────────────────────────────────────────────────────
 
class ChromaBackend:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        try:
            import chromadb
            self._client     = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name     = COLLECTION_NAME,
                metadata = {"hnsw:space": "cosine"},
            )
            self._available = True
        except ImportError:
            self._available = False
 
    @property
    def available(self) -> bool:
        return self._available
 
    def upsert(self, chunks: List[PlaybookChunk], embeddings: np.ndarray) -> None:
        self._collection.upsert(
            ids        = [c.chunk_id for c in chunks],
            embeddings = embeddings.tolist(),
            documents  = [c.text for c in chunks],
            metadatas  = [
                {
                    "playbook_id":      c.playbook_id,
                    "playbook_title":   c.playbook_title,
                    "category":         c.category,
                    "intervention_type":c.intervention_type,
                    "chunk_id":         c.chunk_id,
                    "action_steps":     json.dumps(c.metadata.get("action_steps", [])),
                }
                for c in chunks
            ],
        )
 
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Tuple[dict, float]]:
        res = self._collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results        = min(n_results, self._collection.count()),
            include          = ["documents", "metadatas", "distances"],
        )
        results = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            cosine_score = float(1.0 - dist)   # chroma returns cosine distance
            results.append(({"document": doc, "metadata": meta}, cosine_score))
        return results
 
    def count(self) -> int:
        return self._collection.count()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# FAISS FALLBACK BACKEND
# ─────────────────────────────────────────────────────────────────────────────
 
class FaissBackend:
    def __init__(self):
        self._index    = None
        self._chunks:  List[PlaybookChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._available = False
        try:
            import faiss  # noqa
            self._available = True
        except ImportError:
            pass
 
    @property
    def available(self) -> bool:
        return self._available
 
    def build(self, chunks: List[PlaybookChunk], embeddings: np.ndarray) -> None:
        import faiss
        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatIP(dim)   # inner product = cosine (normalised vecs)
        self._index.add(embeddings.astype(np.float32))
        self._chunks     = chunks
        self._embeddings = embeddings
 
    def save(self, path: Path = FAISS_PATH) -> None:
        import faiss
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".index")))
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks}, f)
 
    def load(self, path: Path = FAISS_PATH) -> bool:
        import faiss
        index_path = path.with_suffix(".index")
        if not path.exists() or not index_path.exists():
            return False
        self._index = faiss.read_index(str(index_path))
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        return True
 
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Tuple[dict, float]]:
        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, min(n_results, len(self._chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append((
                {
                    "document": chunk.text,
                    "metadata": {
                        "playbook_id":       chunk.playbook_id,
                        "playbook_title":    chunk.playbook_title,
                        "category":          chunk.category,
                        "intervention_type": chunk.intervention_type,
                        "chunk_id":          chunk.chunk_id,
                        "action_steps":      json.dumps(chunk.metadata.get("action_steps", [])),
                    },
                },
                float(score),
            ))
        return results
 
    def count(self) -> int:
        return len(self._chunks)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NUMPY PURE FALLBACK (no external vector DB needed)
# ─────────────────────────────────────────────────────────────────────────────
 
class NumpyBackend:
    """Pure NumPy cosine similarity — always available, no extra install."""
 
    def __init__(self):
        self._chunks:    List[PlaybookChunk] = []
        self._embeddings: Optional[np.ndarray] = None
 
    @property
    def available(self) -> bool:
        return True
 
    def build(self, chunks: List[PlaybookChunk], embeddings: np.ndarray) -> None:
        self._chunks     = chunks
        self._embeddings = embeddings   # already L2-normalised
 
    def save(self, path: Path = DATA_DIR / "numpy_index.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "embeddings": self._embeddings}, f)
 
    def load(self, path: Path = DATA_DIR / "numpy_index.pkl") -> bool:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks     = data["chunks"]
        self._embeddings = data["embeddings"]
        return True
 
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Tuple[dict, float]]:
        scores = (self._embeddings @ query_embedding).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_results]
        results = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append((
                {
                    "document": chunk.text,
                    "metadata": {
                        "playbook_id":       chunk.playbook_id,
                        "playbook_title":    chunk.playbook_title,
                        "category":          chunk.category,
                        "intervention_type": chunk.intervention_type,
                        "chunk_id":          chunk.chunk_id,
                        "action_steps":      json.dumps(chunk.metadata.get("action_steps", [])),
                    },
                },
                float(score),
            ))
        return results
 
    def count(self) -> int:
        return len(self._chunks)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────
 
class VectorStore:
    """
    Unified interface over ChromaDB → FAISS → NumPy (in preference order).
    Auto-selects best available backend at build/load time.
    """
 
    def __init__(self, backend, backend_name: str):
        self._backend      = backend
        self.backend_name  = backend_name
 
    # ── Build: embed all chunks and persist ───────────────────────────────────
    @classmethod
    def build(cls, chunks: Optional[List[PlaybookChunk]] = None) -> "VectorStore":
        if chunks is None:
            chunks = get_all_chunks()
 
        print(f"  [VectorStore] Embedding {len(chunks)} playbook chunks …")
        texts      = [c.text for c in chunks]
        embeddings = EmbeddingEngine.encode(texts)
 
        DATA_DIR.mkdir(parents=True, exist_ok=True)
 
        # Try ChromaDB first
        chroma = ChromaBackend()
        if chroma.available:
            chroma.upsert(chunks, embeddings)
            print(f"  [VectorStore] ChromaDB built ({chroma.count()} chunks)")
            return cls(chroma, "chromadb")
 
        # Try FAISS
        faiss_b = FaissBackend()
        if faiss_b.available:
            faiss_b.build(chunks, embeddings)
            faiss_b.save()
            print(f"  [VectorStore] FAISS built ({faiss_b.count()} chunks)")
            return cls(faiss_b, "faiss")
 
        # NumPy fallback
        np_b = NumpyBackend()
        np_b.build(chunks, embeddings)
        np_b.save()
        print(f"  [VectorStore] NumPy index built ({np_b.count()} chunks)")
        return cls(np_b, "numpy")
 
    # ── Load: restore from disk ───────────────────────────────────────────────
    @classmethod
    def load_or_build(cls) -> "VectorStore":
        # ChromaDB: persists automatically
        chroma = ChromaBackend()
        if chroma.available and chroma.count() > 0:
            print(f"  [VectorStore] Loaded ChromaDB ({chroma.count()} chunks)")
            return cls(chroma, "chromadb")
 
        # FAISS
        faiss_b = FaissBackend()
        if faiss_b.available and faiss_b.load():
            print(f"  [VectorStore] Loaded FAISS ({faiss_b.count()} chunks)")
            return cls(faiss_b, "faiss")
 
        # NumPy
        np_b = NumpyBackend()
        if np_b.load():
            print(f"  [VectorStore] Loaded NumPy index ({np_b.count()} chunks)")
            return cls(np_b, "numpy")
 
        # Nothing on disk → build fresh
        return cls.build()
 
    # ── Query interface ───────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        n_results:  int = 5,
        category_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Embed query_text, retrieve top-n chunks, return as RetrievedChunk list.
        Optional category_filter narrows results to matching category.
        """
        q_emb = EmbeddingEngine.encode([query_text])[0]
        # Over-fetch to allow post-filtering
        fetch_n = n_results * 3 if category_filter else n_results
        raw     = self._backend.query(q_emb, n_results=fetch_n)
 
        retrieved: List[RetrievedChunk] = []
        for rank, (item, score) in enumerate(raw, 1):
            meta = item["metadata"]
            if category_filter and meta.get("category", "").lower() != category_filter.lower():
                continue
            action_steps = []
            try:
                action_steps = json.loads(meta.get("action_steps", "[]"))
            except Exception:
                pass
 
            chunk = PlaybookChunk(
                chunk_id          = meta.get("chunk_id", f"chunk_{rank}"),
                playbook_id       = meta.get("playbook_id", ""),
                playbook_title    = meta.get("playbook_title", ""),
                category          = meta.get("category", ""),
                intervention_type = meta.get("intervention_type", ""),
                text              = item["document"],
                metadata          = {"action_steps": action_steps},
            )
            retrieved.append(RetrievedChunk(chunk=chunk, cosine_score=round(score, 4), rank=rank))
            if len(retrieved) >= n_results:
                break
 
        return retrieved
 