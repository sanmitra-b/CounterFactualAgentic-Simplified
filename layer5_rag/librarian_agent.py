from __future__ import annotations

import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_ROOT), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from contracts import KBChunk, LibrarianWorkOrder, LibrarianWorkResult

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = _HERE / "knowledge_base.json"


# ═══════════════════════════════════════════════════════════════════════════════
# TF-IDF RETRIEVER  (pure stdlib — swappable for any vector store)
# ═══════════════════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    """
    Lightweight TF-IDF cosine retriever over a flat JSON knowledge base.

    Each KB document must have:
        id (str), title (str), tags (list[str]), body (str), references (list[str])

    To swap for a vector store, implement a class with the same
    `.retrieve(query: str, k: int) -> list[dict]` signature and pass it to
    LibrarianAgent at construction time.
    """

    def __init__(self, kb_path: str | Path) -> None:
        with open(kb_path, "r", encoding="utf-8") as fh:
            self._docs: list[dict] = json.load(fh)

        # Corpus: title + space-joined tags + body (what gets indexed)
        self._corpus: list[str] = [
            f"{d['title']} {' '.join(d.get('tags', []))} {d['body']}"
            for d in self._docs
        ]
        self._idf = self._build_idf(self._corpus)
        logger.debug("TFIDFRetriever: indexed %d KB documents.", len(self._docs))

    @property
    def size(self) -> int:
        return len(self._docs)

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def _build_idf(self, corpus: list[str]) -> dict[str, float]:
        N = len(corpus)
        df: Counter = Counter()
        for doc in corpus:
            df.update(set(self._tokenise(doc)))
        return {w: math.log((N + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def _tfidf_vec(self, text: str) -> dict[str, float]:
        tokens = self._tokenise(text)
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        return {w: (c / total) * self._idf.get(w, 1.0) for w, c in tf.items()}

    @staticmethod
    def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
        dot   = sum(v1.get(w, 0.0) * s for w, s in v2.items())
        norm1 = math.sqrt(sum(x ** 2 for x in v1.values())) or 1e-9
        norm2 = math.sqrt(sum(x ** 2 for x in v2.values())) or 1e-9
        return dot / (norm1 * norm2)

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Return top-k KB documents scored by TF-IDF cosine similarity."""
        q_vec = self._tfidf_vec(query)
        scored = [
            (self._cosine(q_vec, self._tfidf_vec(text)), doc)
            for doc, text in zip(self._docs, self._corpus)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**doc, "_retrieval_score": round(score, 4)}
            for score, doc in scored[:k]
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# LIBRARIAN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class LibrarianAgent:
    """
    Read-only RAG worker agent.

    Accepts a LibrarianWorkOrder → returns a LibrarianWorkResult.
    Never calls an LLM. Never writes to disk. Never mutates KB state.
    """

    AGENT_NAME       = "LibrarianAgent[Layer5]"
    CONTRACT_VERSION = "1.0"

    def __init__(
        self,
        kb_path: str | Path = DEFAULT_KB_PATH,
        retriever: TFIDFRetriever | None = None,
    ) -> None:
        """
        Pass a custom `retriever` to swap in a vector store.
        If None, the default TFIDFRetriever is used.
        """
        if retriever is not None:
            self._retriever = retriever
        else:
            self._retriever = TFIDFRetriever(kb_path)

        logger.info(
            "%s initialised | KB size: %d docs | Contract v%s",
            self.AGENT_NAME, self._retriever.size, self.CONTRACT_VERSION,
        )

    @property
    def kb_size(self) -> int:
        return self._retriever.size

    def execute(self, order: LibrarianWorkOrder) -> LibrarianWorkResult:
        """
        Main entry point called exclusively by the Layer 5 Supervisor.

        Steps:
          1. Log receipt
          2. Retrieve top-k raw documents from KB
          3. Validate each raw doc → KBChunk contract
          4. Drop invalid chunks; log warnings
          5. Return LibrarianWorkResult

        Never raises — errors go into result.error.
        """
        logger.info(
            "%s | WorkOrder %s | query='%s…' | top_k=%d",
            self.AGENT_NAME,
            order.request_id,
            order.query[:50],
            order.top_k,
        )

        error_messages: list[str] = []
        validated_chunks: list[KBChunk] = []

        # ── Retrieve ──────────────────────────────────────────────────────────
        try:
            raw_chunks: list[dict] = self._retriever.retrieve(
                query=order.query, k=order.top_k,
            )
        except Exception as exc:
            msg = f"Retrieval failed: {exc}"
            logger.error("%s | %s | WorkOrder %s", self.AGENT_NAME, msg, order.request_id)
            return LibrarianWorkResult(
                request_id=order.request_id,
                chunks=[],
                query_used=order.query,
                error=msg,
            )

        # ── Validate each chunk against the KBChunk contract ─────────────────
        for idx, raw in enumerate(raw_chunks):
            try:
                chunk = KBChunk(**raw)
                validated_chunks.append(chunk)
            except ValidationError as val_err:
                msg = (
                    f"KB chunk #{idx + 1} (id='{raw.get('id', '?')}') "
                    f"failed contract validation and was dropped: {val_err}"
                )
                logger.warning("%s | %s", self.AGENT_NAME, msg)
                error_messages.append(msg)

        logger.info(
            "%s | WorkOrder %s complete — %d/%d chunks validated ✓",
            self.AGENT_NAME,
            order.request_id,
            len(validated_chunks),
            len(raw_chunks),
        )

        return LibrarianWorkResult(
            request_id=order.request_id,
            chunks=validated_chunks,
            query_used=order.query,
            error="; ".join(error_messages) if error_messages else None,
        )
