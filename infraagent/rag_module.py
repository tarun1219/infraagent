"""
RAG Module for InfraAgent.

Wraps ChromaDB + sentence-transformers for documentation retrieval.
Provides language-filtered top-k chunk retrieval and query reformulation
from validation error messages.

In stub mode (use_stub=True or sentence-transformers unavailable),
returns empty context without errors — safe for CI/offline use.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


_CHROMA_AVAILABLE = False
_ST_AVAILABLE = False
try:
    import chromadb  # noqa: F401
    _CHROMA_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer as _ST  # noqa: F401
    _ST_AVAILABLE = True
except ImportError:
    pass


class RAGModule:
    """
    Retrieval-Augmented Generation module backed by ChromaDB.

    Args:
        corpus_dir:      Path to corpus directory with markdown files
                         (language sub-directories: kubernetes/, terraform/, dockerfile/).
        chroma_path:     Path to ChromaDB persistence directory.
        top_k:           Number of chunks to return per query.
        embedding_model: sentence-transformers model name.
        use_stub:        If True, skip embedding model load and return empty context.

    Aliases for backward-compat:
        persist_dir → chroma_path
    """

    def __init__(
        self,
        corpus_dir: str = "./rag_corpus",
        chroma_path: str = "./.chroma",
        top_k: int = 5,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_stub: bool = False,
        # backward-compat alias
        persist_dir: str = "",
    ):
        self.top_k           = top_k
        self.corpus_dir      = Path(corpus_dir)
        self.chroma_path     = chroma_path or persist_dir or "./.chroma"
        self.embedding_model = embedding_model
        self._client         = None
        self._ef             = None
        self._stub           = use_stub

        # Validate corpus_dir existence before any further initialisation
        if not use_stub and not self.corpus_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {self.corpus_dir}"
            )

        if not use_stub and _CHROMA_AVAILABLE and _ST_AVAILABLE:
            self._init_chroma()
            self._index_corpus()
        elif not use_stub and not _CHROMA_AVAILABLE:
            # Graceful: no ChromaDB → stub mode
            self._stub = True

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            self._client = chromadb.PersistentClient(path=self.chroma_path)
        except Exception:
            self._stub = True

    def _index_corpus(self) -> None:
        """Walk corpus_dir and upsert markdown files into per-language collections."""
        if self._client is None or not self.corpus_dir.exists():
            if not self.corpus_dir.exists():
                raise FileNotFoundError(
                    f"Corpus directory not found: {self.corpus_dir}"
                )
            return

        for lang_dir in self.corpus_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name.lower()
            collection = self._get_or_create_collection(lang)
            if collection is None:
                continue
            for md_file in sorted(lang_dir.glob("*.md")):
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                # Chunk into ~500-char segments
                chunks = _chunk_text(content, max_len=500)
                if not chunks:
                    continue
                ids  = [f"{md_file.stem}_{i}" for i in range(len(chunks))]
                metas = [{"source": md_file.name, "language": lang}] * len(chunks)
                try:
                    collection.upsert(documents=chunks, ids=ids, metadatas=metas)
                except Exception:
                    pass

    def _get_or_create_collection(self, language: str):
        if self._client is None:
            return None
        try:
            return self._client.get_or_create_collection(
                name=f"iac_{language}",
                embedding_function=self._ef,
            )
        except Exception:
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        language_filter: str = "",
        n_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documentation chunks.

        Returns a list of dicts with keys: ``document``, ``metadata``, ``distance``.
        Returns an empty list in stub mode or when the corpus is empty.
        """
        if self._stub or not query.strip():
            return []

        k = n_results or self.top_k
        lang = language_filter.lower() if language_filter else "kubernetes"
        collection = self._get_or_create_collection(lang)
        if collection is None:
            return []

        try:
            count = collection.count()
            if count == 0:
                return []
            k = min(k, count)
            results   = collection.query(query_texts=[query], n_results=k)
            docs      = results.get("documents",  [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]
            return [
                {"document": d, "metadata": m, "distance": dist}
                for d, m, dist in zip(docs, metas, distances)
            ]
        except Exception:
            return []

    def build_context_string(
        self,
        query: str = "",
        language_filter: str = "",
    ) -> str:
        """
        Retrieve chunks and format them as a prompt-ready context block.

        Returns an empty string when nothing is retrieved.
        """
        chunks = self.retrieve(query=query, language_filter=language_filter)
        if not chunks:
            return ""

        lines = ["## Relevant Documentation\n"]
        for i, chunk in enumerate(chunks, 1):
            doc = chunk.get("document", "")
            src = chunk.get("metadata", {}).get("source", "")
            lines.append(f"### Chunk {i}" + (f" ({src})" if src else ""))
            lines.append(doc.strip())
            lines.append("")
        return "\n".join(lines)

    def reformulate_query(self, errors: List[Dict[str, Any]], base_intent: str = "") -> str:
        """
        Build a RAG query from validation errors for self-correction rounds.

        Combines rule IDs and error messages to focus retrieval on the
        specific misconfiguration types that need to be fixed.
        """
        parts = []
        if base_intent:
            parts.append(base_intent[:200])
        for err in errors[:5]:
            rule_id = err.get("rule_id", "")
            message = err.get("message", "")
            if rule_id:
                parts.append(rule_id)
            if message:
                parts.append(message[:100])
        return " ".join(parts)[:8192]


def _chunk_text(text: str, max_len: int = 500) -> List[str]:
    """Split text into chunks of approximately max_len characters at paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_len:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            current = para[:max_len]
    if current:
        chunks.append(current)
    return chunks
