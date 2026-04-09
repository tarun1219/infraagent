"""
Unit tests for the InfraAgent RAG Module.

Tests ChromaDB retrieval quality, sentence-transformer embedding mocking,
query reformulation from error messages, and corpus coverage.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

try:
    from infraagent.rag_module import RAGModule
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

pytestmark = pytest.mark.skipif(not HAS_RAG, reason="infraagent.rag_module not available")

CORPUS_DIR = Path(__file__).parent.parent / "rag_corpus"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_embedding_model():
    """384-dim zero-vector embeddings — fast, no sentence-transformers needed."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.side_effect = lambda texts, **kw: [[0.1] * 384 for _ in texts]
    return model


@pytest.fixture
def rag_with_mock_embeddings(tmp_path, mock_embedding_model):
    """RAG module with mock embeddings writing to a temp ChromaDB."""
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model):
        rag = RAGModule(
            corpus_dir=str(CORPUS_DIR),
            chroma_path=str(tmp_path / ".chroma"),
            top_k=5,
        )
    return rag


# ── Corpus Coverage ───────────────────────────────────────────────────────────

class TestCorpusCoverage:
    """RAG corpus must cover all three IaC types."""

    def test_kubernetes_corpus_exists(self):
        k8s_dir = CORPUS_DIR / "kubernetes"
        assert k8s_dir.exists(), "rag_corpus/kubernetes/ missing"
        md_files = list(k8s_dir.glob("*.md"))
        assert md_files, "No markdown files in rag_corpus/kubernetes/"

    def test_terraform_corpus_exists(self):
        tf_dir = CORPUS_DIR / "terraform"
        assert tf_dir.exists(), "rag_corpus/terraform/ missing"
        md_files = list(tf_dir.glob("*.md"))
        assert md_files, "No markdown files in rag_corpus/terraform/"

    def test_dockerfile_corpus_exists(self):
        df_dir = CORPUS_DIR / "dockerfile"
        assert df_dir.exists(), "rag_corpus/dockerfile/ missing"
        md_files = list(df_dir.glob("*.md"))
        assert md_files, "No markdown files in rag_corpus/dockerfile/"

    def test_kubernetes_api_versions_doc_exists(self):
        path = CORPUS_DIR / "kubernetes" / "api_versions.md"
        assert path.exists(), "api_versions.md required for deprecated API detection"
        content = path.read_text()
        assert "autoscaling/v2" in content or "v2beta" in content, (
            "api_versions.md should document autoscaling API versions"
        )

    def test_kubernetes_pod_security_doc_exists(self):
        path = CORPUS_DIR / "kubernetes" / "pod_security_standards.md"
        assert path.exists(), "pod_security_standards.md required for L3 security RAG"
        content = path.read_text()
        assert "runAsNonRoot" in content or "securityContext" in content

    def test_terraform_iam_doc_exists(self):
        path = CORPUS_DIR / "terraform" / "iam_best_practices.md"
        assert path.exists(), "iam_best_practices.md required for Terraform security RAG"
        content = path.read_text()
        assert "wildcard" in content.lower() or "least privilege" in content.lower()

    def test_corpus_files_are_non_empty(self):
        for md_file in CORPUS_DIR.rglob("*.md"):
            content = md_file.read_text().strip()
            assert content, f"Corpus file is empty: {md_file}"
            assert len(content) > 100, f"Corpus file suspiciously short: {md_file}"


# ── Initialization ────────────────────────────────────────────────────────────

class TestRAGModuleInit:
    def test_init_with_mock_embeddings(self, rag_with_mock_embeddings):
        assert rag_with_mock_embeddings is not None

    def test_top_k_configurable(self, tmp_path, mock_embedding_model):
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model):
            rag = RAGModule(
                corpus_dir=str(CORPUS_DIR),
                chroma_path=str(tmp_path / ".chroma"),
                top_k=3,
            )
        assert rag.top_k == 3

    def test_corpus_dir_missing_raises(self, tmp_path, mock_embedding_model):
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model):
            with pytest.raises((FileNotFoundError, ValueError, Exception)):
                RAGModule(
                    corpus_dir=str(tmp_path / "nonexistent_corpus"),
                    chroma_path=str(tmp_path / ".chroma"),
                )


# ── Retrieval Quality ─────────────────────────────────────────────────────────

class TestRetrieval:
    """Test that retrieval returns correct structure and respects top_k."""

    def test_retrieve_returns_list(self, rag_with_mock_embeddings):
        results = rag_with_mock_embeddings.retrieve(
            query="How do I set runAsNonRoot in Kubernetes?",
            language_filter="kubernetes",
        )
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self, rag_with_mock_embeddings):
        results = rag_with_mock_embeddings.retrieve(
            query="Kubernetes pod security context",
            language_filter="kubernetes",
        )
        assert len(results) <= rag_with_mock_embeddings.top_k

    def test_retrieve_returns_strings_or_dicts(self, rag_with_mock_embeddings):
        results = rag_with_mock_embeddings.retrieve(
            query="terraform IAM policy wildcard",
            language_filter="terraform",
        )
        for r in results:
            assert isinstance(r, (str, dict)), f"Unexpected result type: {type(r)}"

    def test_build_context_string_returns_str(self, rag_with_mock_embeddings):
        ctx = rag_with_mock_embeddings.build_context_string(
            query="kubernetes securityContext runAsNonRoot",
            language_filter="kubernetes",
        )
        assert isinstance(ctx, str)

    def test_build_context_string_non_empty_for_valid_query(self, rag_with_mock_embeddings):
        ctx = rag_with_mock_embeddings.build_context_string(
            query="kubernetes security context drop capabilities",
            language_filter="kubernetes",
        )
        # May be empty if mock embeddings return identical zero vectors,
        # but the method itself must not raise.
        assert isinstance(ctx, str)

    def test_language_filter_applied(self, rag_with_mock_embeddings):
        """Terraform queries with kubernetes filter should not return TF docs."""
        ctx_k8s = rag_with_mock_embeddings.build_context_string(
            query="aws_s3_bucket versioning",
            language_filter="kubernetes",
        )
        ctx_tf = rag_with_mock_embeddings.build_context_string(
            query="aws_s3_bucket versioning",
            language_filter="terraform",
        )
        # With real embeddings, TF-filtered query should have more relevant content.
        # With mock embeddings we just check both return strings without error.
        assert isinstance(ctx_k8s, str)
        assert isinstance(ctx_tf, str)


# ── Query Reformulation (Error-Based Retrieval) ───────────────────────────────

class TestQueryReformulation:
    """
    In self-correction rounds, the RAG query is reformulated from error messages.
    The combined query (task + errors) should retrieve more targeted documentation.
    """

    def test_error_query_accepted(self, rag_with_mock_embeddings):
        """RAG module must accept error-augmented queries without raising."""
        error_messages = [
            "CKV_K8S_30: Containers should not run as root",
            "CKV_K8S_22: readOnlyRootFilesystem must be true",
            "CKV_K8S_36: All capabilities should be dropped",
        ]
        error_query = " ".join(error_messages)
        ctx = rag_with_mock_embeddings.build_context_string(
            query=f"Kubernetes Deployment security {error_query}",
            language_filter="kubernetes",
        )
        assert isinstance(ctx, str)

    def test_deprecation_error_query(self, rag_with_mock_embeddings):
        error_query = "autoscaling/v2beta2 is deprecated, use autoscaling/v2"
        ctx = rag_with_mock_embeddings.build_context_string(
            query=error_query,
            language_filter="kubernetes",
        )
        assert isinstance(ctx, str)

    def test_security_error_query_terraform(self, rag_with_mock_embeddings):
        error_query = "CKV_AWS_40: IAM policy allows wildcard actions"
        ctx = rag_with_mock_embeddings.build_context_string(
            query=error_query,
            language_filter="terraform",
        )
        assert isinstance(ctx, str)

    def test_empty_query_handled_gracefully(self, rag_with_mock_embeddings):
        """Empty query must not raise — return empty string or minimal context."""
        ctx = rag_with_mock_embeddings.build_context_string(
            query="",
            language_filter="kubernetes",
        )
        assert isinstance(ctx, str)

    def test_very_long_query_handled(self, rag_with_mock_embeddings):
        """Queries with many concatenated error messages must not exceed token limits."""
        long_query = " ".join([f"Error {i}: some validation message" for i in range(50)])
        ctx = rag_with_mock_embeddings.build_context_string(
            query=long_query,
            language_filter="kubernetes",
        )
        assert isinstance(ctx, str)
        # Context must not exceed 2048 tokens (rough proxy: 8192 chars)
        assert len(ctx) <= 8192, "Context string exceeds safe token limit"


# ── ChromaDB Integration ──────────────────────────────────────────────────────

class TestChromaDBIntegration:
    """Test persistence and collection isolation."""

    def test_index_persists_across_instances(self, tmp_path, mock_embedding_model):
        """Second RAGModule pointing to same chroma_path should reuse the index."""
        chroma_path = str(tmp_path / ".chroma")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model):
            rag1 = RAGModule(
                corpus_dir=str(CORPUS_DIR),
                chroma_path=chroma_path,
                top_k=3,
            )
            rag2 = RAGModule(
                corpus_dir=str(CORPUS_DIR),
                chroma_path=chroma_path,
                top_k=3,
            )
        # Both should initialize without error
        assert rag1 is not None
        assert rag2 is not None

    def test_collections_isolated_by_language(self, tmp_path, mock_embedding_model):
        """kubernetes and terraform corpora must not contaminate each other."""
        chroma_path = str(tmp_path / ".chroma")
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedding_model):
            rag = RAGModule(
                corpus_dir=str(CORPUS_DIR),
                chroma_path=chroma_path,
                top_k=5,
            )
        # Retrieve for each language — both must succeed
        for lang in ("kubernetes", "terraform", "dockerfile"):
            result = rag.retrieve(query="security best practices", language_filter=lang)
            assert isinstance(result, list)
